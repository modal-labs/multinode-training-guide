import argparse
import os
import shutil
import tempfile

import torch.distributed as dist

from msswift_mcore_workarounds import load_mcore_checkpoint_lenient


def _build_swift_args(cli_args):
    from swift.megatron.arguments import MegatronExportArguments

    temp_output_dir = tempfile.mkdtemp(prefix="msswift-mcore-export-")
    return MegatronExportArguments(
        model=cli_args.base_model_dir,
        adapters=[cli_args.checkpoint_dir],
        mcore_adapter=cli_args.checkpoint_dir,
        output_dir=temp_output_dir,
        exist_ok=True,
        tuner_type=cli_args.tuner_type,
        merge_lora=True,
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


def export_checkpoint(cli_args) -> None:
    from swift.megatron.model import get_mcore_model
    from swift.pipelines import prepare_model_template
    from swift.megatron.utils import prepare_mcore_model

    swift_args = _build_swift_args(cli_args)
    _, template = prepare_model_template(
        swift_args, load_model=False, download_model=False
    )
    template.set_mode("train")
    template.use_megatron = True
    processor = template.processor
    hf_config = processor.model_info.config
    mg_model = get_mcore_model(swift_args, hf_config)[0]
    bridge = swift_args.megatron_model_meta.bridge_cls(swift_args)

    # The checkpoint directory only contains the LoRA adapter shards. Base model
    # weights still need to come from the original HF model snapshot.
    bridge.load_weights([mg_model], swift_args.model_info.model_dir)
    try:
        peft_model = prepare_mcore_model(swift_args, mg_model)
        bridge.load_weights(
            [mg_model],
            cli_args.checkpoint_dir,
            is_peft_format=True,
        )
        print(f"Loaded HF adapter via GPT bridge from {cli_args.checkpoint_dir}")
    except Exception as hf_adapter_exc:
        print(
            "HF adapter bridge load failed, falling back to MCore adapter load: "
            f"{hf_adapter_exc}"
        )
        peft_model = prepare_mcore_model(swift_args, mg_model)
        load_mcore_checkpoint_lenient(
            swift_args, [peft_model], load_arg="mcore_adapter"
        )
    mg_model = peft_model.merge_and_unload().eval()

    bridge.save_weights(
        [mg_model],
        cli_args.output_dir,
        processor=processor,
        hf_config=hf_config,
    )

    if dist.get_rank() == 0:
        args_path = os.path.join(cli_args.checkpoint_dir, "args.json")
        if os.path.exists(args_path):
            shutil.copy(args_path, os.path.join(cli_args.output_dir, "args.json"))
        print(f"Wrote HF export to {cli_args.output_dir}")

    if dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Custom lenient MCore -> HF export")
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--tuner-type", default="lora")
    parser.add_argument("--padding-free", action="store_true")
    parser.add_argument("--sequence-parallel", action="store_true", default=True)
    parser.add_argument("--decoder-first-pipeline-num-layers", type=int, default=None)
    parser.add_argument("--decoder-last-pipeline-num-layers", type=int, default=None)
    args = parser.parse_args()
    export_checkpoint(args)


if __name__ == "__main__":
    main()
