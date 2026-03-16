import argparse
import os
import tempfile
from typing import Any, Dict, List

from msswift_eval_helpers import read_jsonl, write_json, write_jsonl
from msswift_mcore_workarounds import load_mcore_checkpoint_lenient


def _move_cpu_tensors_to_local_cuda(model) -> None:
    import torch

    target_device = torch.device("cuda", torch.cuda.current_device())
    moved = []
    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            param.data = param.data.to(target_device)
            moved.append(name)
    for name, buffer in model.named_buffers():
        if buffer.device.type == "cpu":
            setattr(
                model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model,
                name.rsplit(".", 1)[-1],
                buffer.to(target_device),
            )
            moved.append(name)
    if moved:
        print(f"Moved CPU tensors to {target_device}: {moved[:32]}")


def _resolve_tokenizer(template) -> Any:
    processor = template.processor
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return processor


def _build_swift_args(cli_args):
    from swift.megatron.arguments import MegatronExportArguments

    temp_output_dir = tempfile.mkdtemp(prefix="msswift-megatron-eval-")
    swift_args = MegatronExportArguments(
        model=cli_args.base_model_dir,
        adapters=[],
        mcore_adapter=cli_args.checkpoint_dir,
        output_dir=temp_output_dir,
        exist_ok=True,
        tuner_type=cli_args.tuner_type,
        merge_lora=cli_args.merge_lora,
        tensor_model_parallel_size=cli_args.tp_size,
        expert_model_parallel_size=cli_args.ep_size,
        pipeline_model_parallel_size=cli_args.pp_size,
        context_parallel_size=cli_args.cp_size,
        sequence_parallel=cli_args.sequence_parallel,
        padding_free=cli_args.padding_free,
        decoder_first_pipeline_num_layers=cli_args.decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers=cli_args.decoder_last_pipeline_num_layers,
        attention_backend="unfused",
        dataset=["placeholder"],
        load_args=True,
    )
    setattr(swift_args, "_checkpoint_dir", cli_args.checkpoint_dir)
    return swift_args


def _load_megatron_model(swift_args):
    from swift.megatron.model import get_mcore_model
    from swift.megatron.utils import prepare_mcore_model
    from swift.pipelines import prepare_model_template

    _, template = prepare_model_template(swift_args, load_model=False, download_model=False)
    template.set_mode("train")
    template.use_megatron = True
    hf_config = template.processor.model_info.config
    mg_model = get_mcore_model(swift_args, hf_config)[0]
    bridge = swift_args.megatron_model_meta.bridge_cls(swift_args)
    bridge.load_weights([mg_model], swift_args.model_info.model_dir)
    if swift_args.tuner_type == "lora":
        peft_model = prepare_mcore_model(swift_args, mg_model)
        try:
            bridge.load_weights(
                [mg_model],
                swift_args._checkpoint_dir,
                is_peft_format=True,
            )
            print(f"Loaded HF adapter via GPT bridge from {swift_args._checkpoint_dir}")
        except Exception as hf_adapter_exc:
            print(
                "HF adapter bridge load failed, falling back to MCore adapter load: "
                f"{hf_adapter_exc}"
            )
            load_mcore_checkpoint_lenient(
                swift_args, [peft_model], load_arg="mcore_adapter"
            )
        base_config = getattr(getattr(peft_model, "base_model", None), "config", None)
        if base_config is not None and not hasattr(base_config, "model_type"):
            hf_model_type = getattr(base_config, "hf_model_type", None)
            if hf_model_type is not None:
                setattr(base_config, "model_type", hf_model_type)
        if swift_args.merge_lora:
            mg_model = peft_model.merge_and_unload()
        else:
            mg_model = peft_model
    _move_cpu_tensors_to_local_cuda(mg_model)
    return mg_model.eval(), template


def _chat_tokens(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> List[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return token_ids


def _score_rows(cli_args) -> None:
    import torch
    import torch.distributed as dist
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
    from swift.megatron.utils import forward_step_helper, get_padding_to
    from swift.megatron.utils.convert_utils import broadcast_mg_logits
    from swift.utils import to_device

    swift_args = _build_swift_args(cli_args)
    mg_model, template = _load_megatron_model(swift_args)
    tokenizer = _resolve_tokenizer(template)
    eval_rows = read_jsonl(cli_args.eval_dataset)
    if cli_args.max_samples is not None:
        eval_rows = eval_rows[: cli_args.max_samples]

    padding_to = get_padding_to(swift_args)
    results = []
    skipped = 0

    for row in eval_rows:
        full_messages = row["messages"] + [
            {"role": "assistant", "content": row["reference_response"]}
        ]
        encoded = template.encode({"messages": full_messages})
        collated = template.data_collator([encoded], padding_to=padding_to)
        collated_input_ids = collated["input_ids"][0].tolist()
        labels = collated["labels"][0].tolist()
        if len(collated_input_ids) < 2 or len(labels) < 2:
            skipped += 1
            continue
        response_mask = [label != -100 for label in labels[1:]]
        if not any(response_mask):
            skipped += 1
            continue

        mg_inputs = to_device(collated, "cuda")
        for key in ["labels", "num_samples", "attention_mask_2d", "text_position_ids"]:
            mg_inputs.pop(key, None)
        mg_inputs["packed_seq_params"] = None

        with torch.inference_mode():
            mg_logits = forward_step_helper(swift_args, mg_model, mg_inputs)
            if (
                swift_args.tensor_model_parallel_size > 1
                and swift_args.task_type != "seq_cls"
                and mg_logits is not None
            ):
                mg_logits = gather_from_tensor_model_parallel_region(mg_logits)

        mg_logits = broadcast_mg_logits(mg_logits)
        if dist.get_rank() != 0:
            continue

        logits = mg_logits[0, : len(collated_input_ids) - 1].float()
        target_ids_list = collated_input_ids[1:]
        target_ids = torch.tensor(
            target_ids_list, device=logits.device, dtype=torch.long
        )
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs_list = token_logprobs.cpu().tolist()
        greedy_all_token_ids = logits.argmax(dim=-1).cpu().tolist()
        response_token_ids = [
            token_id
            for token_id, include in zip(target_ids_list, response_mask)
            if include
        ]
        response_token_logprobs = [
            token_logprob
            for token_logprob, include in zip(token_logprobs_list, response_mask)
            if include
        ]
        greedy_token_ids = [
            token_id
            for token_id, include in zip(greedy_all_token_ids, response_mask)
            if include
        ]

        results.append(
            {
                "id": row["id"],
                "messages": row["messages"],
                "reference_response": row["reference_response"],
                "response_token_ids": response_token_ids,
                "response_token_logprobs": response_token_logprobs,
                "sequence_sum_logprob": sum(response_token_logprobs),
                "sequence_mean_logprob": (
                    sum(response_token_logprobs) / len(response_token_logprobs)
                ),
                "greedy_token_ids": greedy_token_ids,
                "greedy_text": tokenizer.decode(
                    greedy_token_ids, skip_special_tokens=False
                ),
            }
        )

    if dist.get_rank() == 0:
        os.makedirs(cli_args.output_dir, exist_ok=True)
        results_path = os.path.join(cli_args.output_dir, "per_example.jsonl")
        summary_path = os.path.join(cli_args.output_dir, "summary.json")
        write_jsonl(results_path, results)
        write_json(
            summary_path,
            {
                "num_examples": len(results),
                "skipped_examples": skipped,
                "mean_sequence_logprob": (
                    sum(row["sequence_mean_logprob"] for row in results) / len(results)
                    if results
                    else None
                ),
                "results_path": results_path,
            },
        )
        print(f"Wrote Megatron-native eval results to {results_path}")

    if dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Held-out Megatron logprob eval")
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--tuner-type", default="lora")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--padding-free", action="store_true")
    parser.add_argument("--sequence-parallel", action="store_true", default=True)
    parser.add_argument("--decoder-first-pipeline-num-layers", type=int, default=None)
    parser.add_argument("--decoder-last-pipeline-num-layers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    cli_args = parser.parse_args()
    _score_rows(cli_args)


if __name__ == "__main__":
    main()
