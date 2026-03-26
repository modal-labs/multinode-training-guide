"""Modal runtime patches loaded automatically via PYTHONPATH."""


def _log(message: str) -> None:
    print(message, flush=True)


def _register_linear_cross_entropy_module() -> None:
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except Exception as exc:
        _log(
            "[miles-modal] bridge patch unavailable for LinearCrossEntropyModule: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    try:
        AutoMapping.register_module_type("LinearCrossEntropyModule", "column")
    except Exception as exc:
        message = str(exc).lower()
        if any(token in message for token in ("already", "exists", "duplicate")):
            _log(
                "[miles-modal] bridge patch already present for "
                "LinearCrossEntropyModule"
            )
            return
        _log(
            "[miles-modal] bridge patch failed for LinearCrossEntropyModule: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    _log("[miles-modal] registered LinearCrossEntropyModule as column parallel")


def _patch_lora_cpu_serialization() -> None:
    try:
        import base64
        import io
        import torch
        from miles.backends.megatron_utils.update_weight import (
            update_weight_from_tensor as update_weight_mod,
        )
    except Exception as exc:
        _log(
            "[miles-modal] LoRA CPU serialization patch unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    original = getattr(update_weight_mod, "_send_to_colocated_engine", None)
    if original is None:
        _log("[miles-modal] LoRA CPU serialization patch missing target function")
        return

    if getattr(original, "__module__", "") == __name__:
        _log("[miles-modal] LoRA CPU serialization patch already present")
        return

    dist = update_weight_mod.dist
    ray = update_weight_mod.ray
    FlattenedTensorBucket = update_weight_mod.FlattenedTensorBucket
    MultiprocessingSerializer = update_weight_mod.MultiprocessingSerializer

    def _send_to_colocated_engine(
        hf_named_tensors,
        *,
        ipc_engine,
        ipc_gather_src,
        ipc_gather_group,
        weight_version=None,
        lora_config=None,
        lora_name=None,
        lora_loaded=False,
    ):
        # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
        # gather_object is only collective among group members, so we skip entirely.
        if ipc_gather_group is None:
            return [], None

        is_lora = lora_config is not None
        long_live_tensors = []

        if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
        else:
            converted_named_tensors_by_dtypes = {}
            for name, tensor in hf_named_tensors:
                dtype = tensor.dtype
                if dtype not in converted_named_tensors_by_dtypes:
                    converted_named_tensors_by_dtypes[dtype] = []
                converted_named_tensors_by_dtypes[dtype].append((name, tensor))

        serialized_tensors = []
        for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()

            # Modal's colocated LoRA sync can fail on CUDA IPC, and CPU torch.Tensor
            # pickling still goes through multiprocessing resource_sharer. Serialize
            # LoRA flattened buckets into a builtins-only payload so SGLang's safe
            # unpickler can accept it without touching multiprocessing shims.
            if is_lora and isinstance(flattened_tensor, torch.Tensor) and flattened_tensor.is_cuda:
                flattened_tensor = flattened_tensor.detach().cpu()

            if is_lora:
                if not isinstance(flattened_tensor, torch.Tensor):
                    raise TypeError(
                        "Expected LoRA flattened tensor to be a torch.Tensor, got "
                        f"{type(flattened_tensor).__name__}"
                    )
                buffer = io.BytesIO()
                torch.save(flattened_tensor.contiguous(), buffer)
                flattened_tensor_data = {
                    "_miles_modal_format": "torch_save_flattened_lora_v2",
                    "flattened_tensor_torch_save_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "metadata": [
                        {
                            "name": meta.name,
                            "shape": list(meta.shape),
                            "dtype": str(meta.dtype).removeprefix("torch."),
                            "start_idx": meta.start_idx,
                            "end_idx": meta.end_idx,
                            "numel": meta.numel,
                        }
                        for meta in flattened_tensor_bucket.get_metadata()
                    ],
                }
            else:
                flattened_tensor_data = {
                    "flattened_tensor": flattened_tensor,
                    "metadata": flattened_tensor_bucket.get_metadata(),
                }
            long_live_tensors.append(flattened_tensor_data)
            serialized_tensors.append(
                MultiprocessingSerializer.serialize(
                    flattened_tensor_data,
                    output_str=True,
                )
            )

        serialized_named_tensors = (
            [None] * dist.get_world_size(ipc_gather_group)
            if ipc_gather_src == dist.get_rank()
            else None
        )
        dist.gather_object(
            serialized_tensors,
            object_gather_list=serialized_named_tensors,
            dst=ipc_gather_src,
            group=ipc_gather_group,
        )

        refs = []
        if dist.get_rank() == ipc_gather_src:
            if is_lora:
                if lora_loaded:
                    ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))

                refs.append(
                    ipc_engine.load_lora_adapter_from_tensors.remote(
                        lora_name=lora_name,
                        config_dict=lora_config,
                        serialized_tensors=serialized_named_tensors[0][0],
                        load_format="flattened_bucket",
                    )
                )
            else:
                num_dtypes = len(serialized_named_tensors[0])
                for i in range(num_dtypes):
                    kwargs = {
                        "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                        "load_format": "flattened_bucket",
                        "weight_version": str(weight_version),
                    }
                    refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

        return refs, long_live_tensors

    update_weight_mod._send_to_colocated_engine = _send_to_colocated_engine
    _log("[miles-modal] patched colocated LoRA sync to builtins-only flattened buckets")


def _patch_sglang_lora_numpy_rehydration() -> None:
    try:
        import base64
        import io
        import torch
        from sglang.srt.managers import tp_worker as tp_worker_mod
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorMetadata
    except Exception as exc:
        _log(
            "[miles-modal] SGLang LoRA rehydration patch unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    TpModelWorker = getattr(tp_worker_mod, "TpModelWorker", None)
    if TpModelWorker is None:
        _log("[miles-modal] SGLang LoRA rehydration patch missing TpModelWorker")
        return

    original = getattr(TpModelWorker, "load_lora_adapter_from_tensors", None)
    if original is None:
        _log("[miles-modal] SGLang LoRA rehydration patch missing target method")
        return

    if getattr(original, "__module__", "") == __name__:
        _log("[miles-modal] SGLang LoRA rehydration patch already present")
        return

    MultiprocessingSerializer = tp_worker_mod.MultiprocessingSerializer
    FlattenedTensorBucket = tp_worker_mod.FlattenedTensorBucket

    def _torch_dtype_from_name(dtype_name: str):
        return getattr(torch, dtype_name.removeprefix("torch."))

    def load_lora_adapter_from_tensors(self, recv_req):
        # The LoRA code handles TP sharding internally using slice_lora_a_weights
        # and slice_lora_b_weights methods (see lora/layers.py:46-49, mem_pool.py:437-440).
        if recv_req.load_format == "flattened_bucket":
            flattened_data = MultiprocessingSerializer.deserialize(
                recv_req.serialized_tensors
            )
            if flattened_data.get("_miles_modal_format") == "torch_save_flattened_lora_v2":
                raw_bytes = base64.b64decode(flattened_data["flattened_tensor_torch_save_b64"])
                flattened_tensor = torch.load(
                    io.BytesIO(raw_bytes),
                    map_location="cpu",
                )
                metadata = [
                    FlattenedTensorMetadata(
                        name=meta["name"],
                        shape=torch.Size(meta["shape"]),
                        dtype=_torch_dtype_from_name(meta["dtype"]),
                        start_idx=meta["start_idx"],
                        end_idx=meta["end_idx"],
                        numel=meta["numel"],
                    )
                    for meta in flattened_data["metadata"]
                ]
            elif flattened_data.get("_miles_modal_format") == "raw_flattened_lora_v1":
                raw_bytes = base64.b64decode(flattened_data["flattened_tensor_b64"])
                flattened_tensor = torch.frombuffer(
                    memoryview(raw_bytes),
                    dtype=torch.uint8,
                ).clone()
                metadata = [
                    FlattenedTensorMetadata(
                        name=meta["name"],
                        shape=torch.Size(meta["shape"]),
                        dtype=_torch_dtype_from_name(meta["dtype"]),
                        start_idx=meta["start_idx"],
                        end_idx=meta["end_idx"],
                        numel=meta["numel"],
                    )
                    for meta in flattened_data["metadata"]
                ]
            else:
                flattened_tensor = flattened_data["flattened_tensor"]
                metadata = flattened_data["metadata"]
            bucket = FlattenedTensorBucket(
                flattened_tensor=flattened_tensor,
                metadata=metadata,
            )
            tensors = dict(bucket.reconstruct_tensors())
        else:
            tensors = MultiprocessingSerializer.deserialize(recv_req.serialized_tensors)
        result = self.model_runner.load_lora_adapter_from_tensors(
            recv_req.to_ref(),
            tensors,
            recv_req.config_dict,
            recv_req.added_tokens_config,
        )
        return result

    TpModelWorker.load_lora_adapter_from_tensors = load_lora_adapter_from_tensors
    _log("[miles-modal] patched SGLang LoRA load path to rehydrate builtins-only flattened buckets")


def _patch_sglang_logprob_sanitization() -> None:
    try:
        import math
        from sglang.srt.managers import tokenizer_manager as tokenizer_manager_mod
    except Exception as exc:
        _log(
            "[miles-modal] SGLang logprob sanitization patch unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    TokenizerManager = getattr(tokenizer_manager_mod, "TokenizerManager", None)
    if TokenizerManager is None:
        _log("[miles-modal] SGLang logprob sanitization patch missing TokenizerManager")
        return

    original = getattr(TokenizerManager, "detokenize_logprob_tokens", None)
    if original is None:
        _log("[miles-modal] SGLang logprob sanitization patch missing target method")
        return

    if getattr(original, "__module__", "") == __name__:
        _log("[miles-modal] SGLang logprob sanitization patch already present")
        return

    sanitize_state = {"count": 0}

    def _sanitize_logprob(value):
        try:
            numeric = float(value)
        except Exception:
            return value

        if math.isnan(numeric) or math.isinf(numeric):
            sanitized = 0.0
        elif numeric > 0.0:
            sanitized = 0.0
        else:
            sanitized = numeric

        if sanitized != numeric:
            sanitize_state["count"] += 1
            if sanitize_state["count"] <= 8:
                _log(
                    "[miles-modal] sanitized SGLang logprob "
                    f"{numeric!r} -> {sanitized!r}"
                )
        return sanitized

    def detokenize_logprob_tokens(self, token_logprobs_val, token_logprobs_idx, decode_to_text):
        sanitized_vals = [_sanitize_logprob(value) for value in token_logprobs_val]
        return original(self, sanitized_vals, token_logprobs_idx, decode_to_text)

    TokenizerManager.detokenize_logprob_tokens = detokenize_logprob_tokens
    _log("[miles-modal] patched SGLang logprob detokenization to sanitize non-finite values")


def _patch_sglang_sampling_probability_sanitization() -> None:
    try:
        import torch
        from sglang.srt.layers import sampler as sampler_mod
    except Exception as exc:
        _log(
            "[miles-modal] SGLang sampling probability patch unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    original = getattr(sampler_mod, "sampling_from_probs_torch", None)
    if original is None:
        _log("[miles-modal] SGLang sampling probability patch missing target function")
        return

    if getattr(original, "__module__", "") == __name__:
        _log("[miles-modal] SGLang sampling probability patch already present")
        return

    sanitize_state = {"count": 0}

    def _sanitize_probs(probs: torch.Tensor) -> torch.Tensor:
        probs_fp32 = probs.float()
        valid_mask = torch.isfinite(probs_fp32) & (probs_fp32 >= 0)
        safe_probs = torch.where(valid_mask, probs_fp32, torch.zeros_like(probs_fp32))
        row_sums = safe_probs.sum(dim=-1, keepdim=True)
        zero_rows = row_sums <= 0

        has_invalid = bool((~valid_mask).any().item())
        has_zero_rows = bool(zero_rows.any().item())

        if has_zero_rows:
            fallback_scores = torch.nan_to_num(
                probs_fp32,
                nan=float("-inf"),
                posinf=float("-inf"),
                neginf=float("-inf"),
            )
            fallback_indices = fallback_scores.argmax(dim=-1, keepdim=True)
            fallback_probs = torch.zeros_like(safe_probs)
            fallback_probs.scatter_(-1, fallback_indices, 1.0)
            safe_probs = torch.where(zero_rows, fallback_probs, safe_probs)
            row_sums = safe_probs.sum(dim=-1, keepdim=True)

        if has_invalid or has_zero_rows:
            sanitize_state["count"] += 1
            if sanitize_state["count"] <= 8:
                _log(
                    "[miles-modal] sanitized SGLang sampling probs "
                    f"(invalid_entries={int((~valid_mask).sum().item())}, "
                    f"zero_rows={int(zero_rows.sum().item())})"
                )

        return safe_probs / row_sums.clamp_min(1e-12)

    def sampling_from_probs_torch(
        probs: torch.Tensor,
        sampling_seed=None,
        positions=None,
    ):
        safe_probs = _sanitize_probs(probs)
        return original(
            safe_probs,
            sampling_seed=sampling_seed,
            positions=positions,
        )

    sampler_mod.sampling_from_probs_torch = sampling_from_probs_torch
    _log("[miles-modal] patched SGLang sampling to sanitize invalid probability rows")


_register_linear_cross_entropy_module()
_patch_lora_cpu_serialization()
_patch_sglang_lora_numpy_rehydration()
_patch_sglang_logprob_sanitization()
_patch_sglang_sampling_probability_sanitization()
