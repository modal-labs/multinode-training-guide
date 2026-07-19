# pyright: reportMissingImports=false
"""Finalize and verify DeepSeek-V4-Flash pipeline-parallel LoRA checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HF_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
MODEL_LAYERS = 43
LORA_RANK = 64
LORA_ALPHA = 64
PIPELINE_STAGES = 4
CHECKPOINT_STEP = 4

TARGET_MAP = {
    "wq_a": "q_a_proj",
    "wq_b": "q_b_proj",
    "wkv": "kv_proj",
}
TARGET_SHAPES = {
    "wq_a": {
        "lora_A": (LORA_RANK, 4096),
        "lora_B": (1024, LORA_RANK),
    },
    "wq_b": {
        "lora_A": (LORA_RANK, 1024),
        "lora_B": (32768, LORA_RANK),
    },
    "wkv": {
        "lora_A": (LORA_RANK, 4096),
        "lora_B": (512, LORA_RANK),
    },
}
PEFT_TARGET_PATTERN = r"model\.layers\.\d+\.self_attn\.(q_a_proj|q_b_proj|kv_proj)"
EXPECTED_TENSOR_COUNT = MODEL_LAYERS * len(TARGET_MAP) * 2


def _validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")


def finalized_adapter(
    checkpoints_dir: str | Path,
    run_id: str,
) -> tuple[Path, dict[str, object]]:
    """Return a finalized adapter after verifying its manifest and digest."""
    _validate_run_id(run_id)
    checkpoint_dir = Path(checkpoints_dir) / run_id / f"epoch_0_step_{CHECKPOINT_STEP}"
    model_dir = checkpoint_dir / "model"
    adapter_path = model_dir / "adapter_model.safetensors"
    config_path = model_dir / "adapter_config.json"
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    for path in (adapter_path, config_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing finalized adapter artifact: {path}")

    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("run_id") != run_id
        or manifest.get("step") != CHECKPOINT_STEP
        or manifest.get("adapter_tensor_count") != EXPECTED_TENSOR_COUNT
    ):
        raise RuntimeError(
            f"Checkpoint manifest does not match run {run_id!r}: {manifest}"
        )

    digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    if digest != manifest.get("adapter_sha256"):
        raise RuntimeError(
            f"Adapter SHA-256 {digest} does not match {manifest.get('adapter_sha256')}"
        )

    adapter_config = json.loads(config_path.read_text())
    if (
        adapter_config.get("r") != LORA_RANK
        or adapter_config.get("lora_alpha") != LORA_ALPHA
    ):
        raise RuntimeError(
            f"Expected rank-{LORA_RANK}/alpha-{LORA_ALPHA}, found "
            f"r={adapter_config.get('r')}, "
            f"alpha={adapter_config.get('lora_alpha')}"
        )
    return model_dir, manifest


def _validate_with_peft(model_dir: Path) -> dict[str, object]:
    """Load every adapter tensor through PEFT without materializing the base."""
    import torch
    from accelerate import init_empty_weights
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import (
        load_peft_weights,
        set_peft_model_state_dict,
    )
    from transformers import AutoConfig, AutoModelForCausalLM

    base_config = AutoConfig.from_pretrained(HF_MODEL, trust_remote_code=False)
    adapter_config = PeftConfig.from_pretrained(str(model_dir))
    with init_empty_weights(include_buffers=True):
        base_model = AutoModelForCausalLM.from_config(
            base_config,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
    model = get_peft_model(base_model, adapter_config, low_cpu_mem_usage=True)
    adapter_params = [
        (name, param) for name, param in model.named_parameters() if "lora_" in name
    ]
    if len(adapter_params) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"PEFT injected {len(adapter_params)} adapter parameters, expected "
            f"{EXPECTED_TENSOR_COUNT}"
        )

    for module in model.modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            module.lora_A["default"].to_empty(device="cpu")
            module.lora_B["default"].to_empty(device="cpu")

    state = load_peft_weights(str(model_dir), device="cpu")
    load_result = set_peft_model_state_dict(model, state, adapter_name="default")
    missing = [key for key in load_result.missing_keys if "lora_" in key]
    if missing:
        raise RuntimeError(f"PEFT load is missing adapter keys: {missing[:10]}")
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"PEFT load has unexpected keys: {load_result.unexpected_keys[:10]}"
        )

    adapter_params = [
        (name, param) for name, param in model.named_parameters() if "lora_" in name
    ]
    invalid = [
        name
        for name, param in adapter_params
        if param.is_meta
        or not torch.isfinite(param).all()
        or (".lora_B." in name and torch.count_nonzero(param).item() == 0)
    ]
    if invalid:
        raise RuntimeError(f"PEFT loaded invalid adapter parameters: {invalid[:10]}")

    layers = sorted(
        {
            int(match.group(1))
            for name, _ in adapter_params
            if (match := re.search(r"\.layers\.(\d+)\.", name))
        }
    )
    targets = sorted(
        {
            match.group(1)
            for name, _ in adapter_params
            if (
                match := re.search(
                    r"\.self_attn\.(q_a_proj|q_b_proj|kv_proj)\.",
                    name,
                )
            )
        }
    )
    if layers != list(range(MODEL_LAYERS)):
        raise RuntimeError(f"PEFT load covered unexpected layers: {layers}")
    if targets != ["kv_proj", "q_a_proj", "q_b_proj"]:
        raise RuntimeError(f"PEFT load covered unexpected targets: {targets}")

    return {
        "adapter_parameters": len(adapter_params),
        "layers": len(layers),
        "targets": targets,
    }


def finalize_adapter(
    checkpoints_dir: str | Path,
    run_id: str,
) -> dict[str, object]:
    """Merge PP-local shards into one PEFT adapter and verify it."""
    import torch
    from safetensors.torch import load_file, save_file

    _validate_run_id(run_id)
    run_dir = Path(checkpoints_dir) / run_id
    candidates: list[tuple[tuple[int, int], Path]] = []
    checkpoint_pattern = re.compile(r"epoch_(\d+)_step_(\d+)")
    if run_dir.exists():
        for path in run_dir.iterdir():
            match = checkpoint_pattern.fullmatch(path.name)
            if path.is_dir() and match:
                candidates.append(((int(match.group(1)), int(match.group(2))), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint steps found under {run_dir}")

    (epoch, step), checkpoint_dir = max(candidates)
    if step != CHECKPOINT_STEP:
        raise RuntimeError(
            f"Latest checkpoint is step {step}, expected {CHECKPOINT_STEP}"
        )

    model_dir = checkpoint_dir / "model"
    stage_shards = sorted(model_dir.glob("adapter_model.pp-*-of-*.safetensors"))
    if len(stage_shards) != PIPELINE_STAGES:
        raise RuntimeError(
            f"Found {len(stage_shards)} PP shards, expected {PIPELINE_STAGES}"
        )

    merged: dict[str, torch.Tensor] = {}
    for shard_path in stage_shards:
        shard = load_file(str(shard_path), device="cpu")
        duplicate_keys = sorted(set(merged).intersection(shard))
        if duplicate_keys:
            raise RuntimeError(
                f"Duplicate keys in {shard_path.name}: {duplicate_keys[:10]}"
            )
        nonfinite = [
            name for name, tensor in shard.items() if not torch.isfinite(tensor).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"Non-finite tensors in {shard_path.name}: {nonfinite[:10]}"
            )
        merged.update(shard)

    if len(merged) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"Merged adapter has {len(merged)} tensors, "
            f"expected {EXPECTED_TENSOR_COUNT}"
        )

    expected_adapters = ("lora_A", "lora_B")
    matched_names: set[str] = set()
    peft_state: dict[str, torch.Tensor] = {}
    for layer in range(MODEL_LAYERS):
        for source_target, peft_target in TARGET_MAP.items():
            for adapter in expected_adapters:
                needle = f".layers.{layer}.self_attn.{source_target}.{adapter}.weight"
                matches = [key for key in merged if needle in key]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one tensor matching {needle}, found {len(matches)}"
                    )

                name = matches[0]
                tensor = merged[name]
                expected_shape = TARGET_SHAPES[source_target][adapter]
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(
                        f"{name} has shape {tuple(tensor.shape)}, "
                        f"expected {expected_shape}"
                    )
                if adapter == "lora_B" and torch.count_nonzero(tensor).item() == 0:
                    raise RuntimeError(f"Untrained LoRA-B tensor: {name}")

                matched_names.add(name)
                peft_name = name.replace(
                    f".self_attn.{source_target}.",
                    f".self_attn.{peft_target}.",
                    1,
                )
                if peft_name in peft_state:
                    raise RuntimeError(f"Duplicate PEFT key: {peft_name}")
                peft_state[peft_name] = tensor

    unexpected = sorted(set(merged).difference(matched_names))
    if unexpected:
        raise RuntimeError(f"Unexpected adapter tensors: {unexpected[:10]}")

    adapter_config_path = model_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT config: {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text())
    if (
        adapter_config.get("r") != LORA_RANK
        or adapter_config.get("lora_alpha") != LORA_ALPHA
    ):
        raise RuntimeError(
            f"Expected rank-{LORA_RANK}/alpha-{LORA_ALPHA}, found "
            f"r={adapter_config.get('r')}, "
            f"alpha={adapter_config.get('lora_alpha')}"
        )
    adapter_config["target_modules"] = PEFT_TARGET_PATTERN

    temporary_config = adapter_config_path.with_suffix(".json.tmp")
    temporary_config.write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n"
    )
    temporary_config.replace(adapter_config_path)

    adapter_path = model_dir / "adapter_model.safetensors"
    temporary_adapter = adapter_path.with_suffix(".safetensors.tmp")
    save_file(peft_state, str(temporary_adapter))
    temporary_adapter.replace(adapter_path)

    peft_validation = _validate_with_peft(model_dir)
    manifest = {
        "run_id": run_id,
        "epoch": epoch,
        "step": step,
        "pipeline_stages": PIPELINE_STAGES,
        "adapter_tensor_count": len(peft_state),
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "layers": MODEL_LAYERS,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "targets": list(TARGET_MAP.values()),
        "peft_validation": peft_validation,
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
