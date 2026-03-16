import argparse
import os
import shutil
import tempfile
import types

import torch.distributed as dist

from msswift_mcore_workarounds import load_mcore_checkpoint_lenient


def _build_swift_args(cli_args):
    from swift.megatron.arguments import MegatronExportArguments

    temp_output_dir = tempfile.mkdtemp(prefix="msswift-mcore-export-")
    swift_args = MegatronExportArguments(
        model=cli_args.base_model_dir,
        adapters=[],
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
    setattr(swift_args, "_checkpoint_dir", cli_args.checkpoint_dir)
    return swift_args


def _patch_bridge_for_pipeline_gaps(bridge) -> None:
    original_set_layer_state = bridge._set_layer_state
    original_set_layer_attn = bridge._set_layer_attn
    original_set_layer_mlp = bridge._set_layer_mlp

    def _set_layer_state_safe(self, mg_layer, hf_state_dict, hf_prefix, layer_idx, to_mcore):
        if mg_layer is None:
            return {}
        return original_set_layer_state(mg_layer, hf_state_dict, hf_prefix, layer_idx, to_mcore)

    def _set_layer_attn_safe(self, mg_layer, hf_state_dict, layer_idx, to_mcore):
        if mg_layer is None or getattr(mg_layer, "self_attention", None) is None:
            return {}
        return original_set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore)

    def _set_layer_mlp_safe(self, mg_layer, hf_state_dict, layer_idx, to_mcore):
        if mg_layer is None or getattr(mg_layer, "mlp", None) is None:
            return {}
        return original_set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore)

    bridge._set_layer_state = types.MethodType(_set_layer_state_safe, bridge)
    bridge._set_layer_attn = types.MethodType(_set_layer_attn_safe, bridge)
    bridge._set_layer_mlp = types.MethodType(_set_layer_mlp_safe, bridge)


def _patch_bridge_for_virtual_pipeline_exhaustion(bridge) -> None:
    original_convert = bridge._convert

    def _convert_safe(self, *args, **kwargs):
        if len(args) < 4:
            yield from original_convert(*args, **kwargs)
            return

        mg_models, hf_state_dict, hf_prefix, to_mcore = args[:4]
        tqdm_desc = kwargs.get("tqdm_desc", "Converting: ")
        g = original_convert.__globals__
        torch_mod = g["torch"]
        dist_mod = g["dist"]
        mpu = g["mpu"]
        is_master = g["is_master"]
        tqdm = g["tqdm"]
        mcore_013 = g["mcore_013"]

        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
            hf_state_dict = self._convert_hf_state_dict(hf_state_dict, to_mcore)
        else:
            hf_state_dict = {}

        mg_models_iter = iter(mg_models)
        mg_model = next(mg_models_iter)
        if mcore_013:
            is_pp_first_stage = mpu.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=mg_model.vp_stage
            )
            is_pp_last_stage = mpu.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=mg_model.vp_stage
            )
        else:
            is_pp_first_stage = mpu.is_pipeline_first_stage()
            is_pp_last_stage = mpu.is_pipeline_last_stage()

        if not to_mcore or is_pp_first_stage:
            hf_state_dict.update(
                self._convert_pre_process(mg_model, hf_state_dict, "", to_mcore)
            )
        if to_mcore:
            yield
        else:
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())
            hf_state_dict = {}

        layer_idx = 0
        disable_tqdm = self._disable_tqdm or not is_master()
        prog_bar = tqdm(
            range(self.config.num_layers),
            dynamic_ncols=True,
            desc=tqdm_desc,
            disable=disable_tqdm,
        )
        exhausted_virtual_models = False
        while layer_idx < self.config.num_layers:
            lm_model = (
                getattr(mg_model, "language_model") if self.is_multimodal else mg_model
            )
            if len(lm_model.decoder.layers) > 0:
                start_idx = lm_model.decoder.layers[0].layer_number - 1
                mg_layer_available = (
                    start_idx <= layer_idx < lm_model.decoder.layers[-1].layer_number
                )
            else:
                mg_layer_available = False
            if mg_layer_available:
                mg_layer = lm_model.decoder.layers[layer_idx - start_idx]
            else:
                if to_mcore:
                    layer_idx += 1
                    prog_bar.update()
                    continue
                mg_layer = None
            if not to_mcore and self.pp_size > 1:
                has_model = torch_mod.tensor(
                    [mg_layer is not None], dtype=torch_mod.bool, device="cuda"
                )
                dist_mod.all_reduce(has_model, group=self.pp_group)
                if not has_model:
                    try:
                        mg_model = next(mg_models_iter)  # compat vpp
                        continue
                    except StopIteration:
                        exhausted_virtual_models = True
                        layer_idx += 1
                        prog_bar.update()
                        continue
            res = self._set_layer_state(
                mg_layer, hf_state_dict, f"{self.hf_layers_prefix}.", layer_idx, to_mcore
            )
            layer_idx += 1
            prog_bar.update()
            if to_mcore:
                yield
            else:
                yield from list(self._add_prefix(res, hf_prefix).items())
                hf_state_dict = {}

        if exhausted_virtual_models:
            return

        if (not to_mcore or is_pp_last_stage) and self.config.mtp_num_layers:
            lm_model = (
                getattr(mg_model, "language_model") if self.is_multimodal else mg_model
            )
            if to_mcore and self.pp_rank > 0:
                self._set_state_dict(
                    lm_model,
                    "embedding.word_embeddings.weight",
                    hf_state_dict,
                    self.hf_embed_key,
                    to_mcore,
                )
            layer_idx = 0
            while layer_idx < self.config.mtp_num_layers:
                res = self._convert_mtp_layer(
                    lm_model, hf_state_dict, f"{self.hf_mtp_prefix}.", layer_idx, to_mcore
                )
                layer_idx += 1
                if to_mcore:
                    yield
                else:
                    yield from list(self._add_prefix(res, hf_prefix).items())
                    hf_state_dict = {}
        if not to_mcore or is_pp_last_stage:
            hf_state_dict.update(
                self._convert_post_process(mg_model, hf_state_dict, "", to_mcore)
            )
        if to_mcore:
            yield
        else:
            hf_state_dict = self._convert_hf_state_dict(hf_state_dict, to_mcore)
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())

    bridge._convert = types.MethodType(_convert_safe, bridge)


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
    _patch_bridge_for_pipeline_gaps(bridge)
    _patch_bridge_for_virtual_pipeline_exhaustion(bridge)

    # The checkpoint directory only contains the LoRA adapter shards. Base model
    # weights still need to come from the original HF model snapshot.
    bridge.load_weights([mg_model], swift_args.model_info.model_dir)
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
    export_models = [mg_model]
    if not cli_args.peft_format:
        export_models = [peft_model.merge_and_unload().eval()]

    bridge.save_weights(
        export_models,
        cli_args.output_dir,
        is_peft_format=cli_args.peft_format,
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
    parser.add_argument("--peft-format", action="store_true")
    args = parser.parse_args()
    export_checkpoint(args)


if __name__ == "__main__":
    main()
