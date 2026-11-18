"""Runtime compatibility patches for Modal-hosted SLIME jobs."""

from __future__ import annotations

import io
import os


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off", ""}


def _log_patch(message: str) -> None:
    if _env_enabled("MODAL_RUNTIME_PATCHES_LOG", default=True):
        print(f"Modal runtime patch: {message}", flush=True)


def _clear_dsa_context_parallel_hint(config, source: str) -> None:
    if (
        getattr(config, "experimental_attention_variant", None) == "dsa"
        and getattr(config, "context_parallel_size", 1) != 1
    ):
        config.experimental_attention_variant = None
        _log_patch(f"cleared DSAttention CP guard before {source}")


def _patch_glm_moe_dsa_hf_config(hf_config) -> None:
    if getattr(hf_config, "model_type", None) != "glm_moe_dsa":
        return

    override = os.environ.get("GLM_MOE_DSA_QK_POS_EMB_HEAD_DIM", "64")
    try:
        qk_pos_emb_head_dim = int(override)
    except ValueError:
        return

    current = getattr(hf_config, "qk_rope_head_dim", None)
    if current == qk_pos_emb_head_dim:
        return

    hf_config.qk_rope_head_dim = qk_pos_emb_head_dim
    _log_patch(
        "corrected GLM qk_rope_head_dim "
        f"from {current} to {qk_pos_emb_head_dim}"
    )


def _patch_glm_moe_dsa_bridge_config(config, source: str) -> None:
    override = os.environ.get("GLM_MOE_DSA_QK_POS_EMB_HEAD_DIM", "64")
    try:
        qk_pos_emb_head_dim = int(override)
    except ValueError:
        return

    if getattr(config, "qk_pos_emb_head_dim", None) == qk_pos_emb_head_dim:
        return

    # GLM-5.2-FP8's HF config reports qk_rope_head_dim=192, but its weights
    # encode qk_nope=192 and qk_rope=64:
    #   q_b_proj: 64 * (192 + 64)
    #   kv_a_proj_with_mqa: 512 + 64
    # Patch the deferred bridge config object because the training load path
    # does not necessarily use Slime's plugin bridge wrapper.
    if not (
        getattr(config, "hidden_size", None) == 6144
        and getattr(config, "num_layers", None) == 78
        and getattr(config, "kv_lora_rank", None) == 512
        and getattr(config, "qk_head_dim", None) == 192
    ):
        return

    current = getattr(config, "qk_pos_emb_head_dim", None)
    config.qk_pos_emb_head_dim = qk_pos_emb_head_dim
    _log_patch(
        "corrected GLM bridge qk_pos_emb_head_dim "
        f"from {current} to {qk_pos_emb_head_dim} before {source}"
    )


def _patch_fp8_dequant_cuda() -> None:
    if not _env_enabled("PATCH_FP8_DEQUANT_CUDA", default=True):
        return

    try:
        import torch
        import mbridge.models.ext.deepseek_v3.dequant_fp8_safetensor_io as fp8_io
    except Exception:
        return

    original = fp8_io.weight_dequant
    if getattr(original, "_modal_cuda_dequant_patch", False):
        return

    def weight_dequant_cuda(weight, scale_inv, *args, **kwargs):
        device = torch.device("cuda", torch.cuda.current_device())
        weight = weight.to(device=device, non_blocking=True)
        scale_inv = scale_inv.to(device=device, non_blocking=True)
        return original(weight.contiguous(), scale_inv.contiguous(), *args, **kwargs)

    weight_dequant_cuda._modal_cuda_dequant_patch = True
    fp8_io.weight_dequant = weight_dequant_cuda


def _patch_dcp_thread_count() -> None:
    thread_count = os.environ.get("MODAL_DCP_THREAD_COUNT")
    if not thread_count:
        return

    try:
        forced_thread_count = int(thread_count)
    except ValueError:
        return
    if forced_thread_count < 1:
        return

    try:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async
    except Exception:
        return

    writer_cls = fs_async.FileSystemWriterAsync
    if getattr(writer_cls, "_modal_thread_count_patch", False):
        return

    original_init = writer_cls.__init__

    def init_with_thread_count(self, path, *args, **kwargs):
        kwargs["thread_count"] = forced_thread_count
        return original_init(self, path, *args, **kwargs)

    init_with_thread_count._modal_original_init = original_init
    writer_cls.__init__ = init_with_thread_count
    writer_cls._modal_thread_count_patch = True


def _patch_dcp_buffered_torch_save() -> None:
    if not _env_enabled("MODAL_DCP_BUFFERED_TORCH_SAVE"):
        return

    try:
        import torch
        import torch.distributed.checkpoint.filesystem as dcp_fs
        import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async
        from torch.distributed.checkpoint.planner import WriteItemType
        from torch.distributed.checkpoint.storage import WriteResult
    except Exception:
        return

    if getattr(fs_async, "_modal_buffered_torch_save_patch", False):
        return

    original_write_item = fs_async._write_item

    def write_item_buffered(
        transforms,
        stream,
        data,
        write_item,
        storage_key,
        serialization_format=dcp_fs.SerializationFormat.TORCH_SAVE,
    ):
        if (
            serialization_format != dcp_fs.SerializationFormat.TORCH_SAVE
            or write_item.type == WriteItemType.BYTE_IO
            or not isinstance(data, torch.Tensor)
            or data.device.type != "cpu"
        ):
            return original_write_item(
                transforms,
                stream,
                data,
                write_item,
                storage_key,
                serialization_format=serialization_format,
            )

        offset = stream.tell()
        transform_to, transform_descriptors = transforms.transform_save_stream(
            write_item, stream
        )
        try:
            buffer = io.BytesIO()
            torch.save(data, buffer)
            transform_to.write(buffer.getbuffer())
            transform_to.close()
        except Exception:
            try:
                transform_to.close()
            except Exception:
                pass
            raise

        length = stream.tell() - offset
        if len(transform_descriptors) == 0:
            transform_descriptors = None

        return WriteResult(
            index=write_item.index,
            size_in_bytes=length,
            storage_data=dcp_fs._StorageInfo(
                storage_key,
                offset,
                length,
                transform_descriptors=transform_descriptors,
            ),
        )

    fs_async._write_item = write_item_buffered
    fs_async._modal_buffered_torch_save_patch = True


def _patch_flattened_tensor_bucket_single_tensor_zero_copy() -> None:
    if not _env_enabled("PATCH_SINGLE_TENSOR_BUCKET_ZERO_COPY", default=True):
        return

    try:
        import torch
        import sglang.srt.weight_sync.tensor_bucket as tensor_bucket
    except Exception:
        return

    bucket_cls = tensor_bucket.FlattenedTensorBucket
    if getattr(bucket_cls, "_modal_single_tensor_zero_copy_patch", False):
        return

    original_init = bucket_cls.__init__
    metadata_cls = tensor_bucket.FlattenedTensorMetadata

    def init_single_tensor_zero_copy(
        self,
        named_tensors=None,
        flattened_tensor=None,
        metadata=None,
    ):
        if named_tensors is None or len(named_tensors) != 1:
            return original_init(
                self,
                named_tensors=named_tensors,
                flattened_tensor=flattened_tensor,
                metadata=metadata,
            )

        name, tensor = named_tensors[0]
        flattened = tensor.flatten().view(torch.uint8)
        numel = flattened.numel()
        self.metadata = [
            metadata_cls(
                name=name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                start_idx=0,
                end_idx=numel,
                numel=numel,
            )
        ]
        self.flattened_tensor = flattened

    bucket_cls.__init__ = init_single_tensor_zero_copy
    bucket_cls._modal_single_tensor_zero_copy_patch = True
    _log_patch("patched FlattenedTensorBucket single-tensor zero-copy path")


def _patch_dsa_kv_pool_cpu_copy_signature() -> None:
    if not _env_enabled("PATCH_DSA_KV_POOL_MAMBA_INDICES", default=True):
        return

    try:
        import importlib
        import inspect
    except Exception:
        return

    module_names = (
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.mem_cache.allocator.base",
        "sglang.srt.mem_cache.allocator.paged",
    )
    patched = []

    def wrap_method(pool_cls, method_name: str) -> None:
        original = getattr(pool_cls, method_name, None)
        if original is None or getattr(original, "_modal_mamba_indices_patch", False):
            return

        try:
            if "mamba_indices" in inspect.signature(original).parameters:
                return
        except Exception:
            pass

        def method_ignoring_mamba_indices(
            self, *args, __original=original, mamba_indices=None, **kwargs
        ):
            return __original(self, *args, **kwargs)

        method_ignoring_mamba_indices._modal_mamba_indices_patch = True
        method_ignoring_mamba_indices._modal_original = original
        setattr(pool_cls, method_name, method_ignoring_mamba_indices)
        patched.append(f"{pool_cls.__module__}.{pool_cls.__name__}.{method_name}")

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for class_name, pool_cls in vars(module).items():
            if not isinstance(pool_cls, type):
                continue
            if class_name not in {"DSATokenToKVPool", "NSATokenToKVPool"}:
                continue
            wrap_method(pool_cls, "get_cpu_copy")
            wrap_method(pool_cls, "load_cpu_copy")

    if patched:
        _log_patch(
            "patched KV pool CPU-copy mamba_indices compatibility for "
            + ", ".join(patched)
        )


def _patch_glm_moe_dsa_bridge_context_parallel() -> None:
    """Let Slime's GLM5 DSA spec own CP instead of Megatron-Core DSAttention."""
    if not _env_enabled("PATCH_GLM_MOE_DSA_CP", default=True):
        return

    _patch_megatron_core_dsa_context_parallel_guard()
    _patch_megatron_bridge_dsa_context_parallel_guard()

    try:
        import slime_plugins.mbridge.deepseek_v32 as deepseek_v32
    except Exception:
        return

    bridge_cls = deepseek_v32.DeepseekV32Bridge
    if getattr(bridge_cls, "_modal_glm_moe_dsa_cp_patch", False):
        return

    original_build_config = bridge_cls._build_config

    def build_config_without_core_dsa(self, *args, **kwargs):
        _patch_glm_moe_dsa_hf_config(self.hf_config)
        config = original_build_config(self, *args, **kwargs)
        if getattr(self.hf_config, "model_type", None) == "glm_moe_dsa":
            # Slime's glm5 spec replaces the layer attention module with its own
            # DSAMLASelfAttention. Leaving this bridge flag as "dsa" sends the
            # provider through Megatron-Core's DSAttention validation, which
            # currently rejects context parallelism before Slime can replace it.
            _clear_dsa_context_parallel_hint(config, "DeepseekV32Bridge._build_config")
        return config

    bridge_cls._build_config = build_config_without_core_dsa
    bridge_cls._modal_glm_moe_dsa_cp_patch = True
    _log_patch("patched DeepseekV32Bridge._build_config")


def _patch_megatron_bridge_dsa_context_parallel_guard() -> None:
    try:
        import megatron.bridge.models.transformer_config as bridge_transformer_config
    except Exception:
        return

    for class_name in ("TransformerConfig", "MLATransformerConfig"):
        config_cls = getattr(bridge_transformer_config, class_name, None)
        if (
            config_cls is None
            or config_cls.__dict__.get("_modal_glm_moe_dsa_finalize_patch", False)
        ):
            continue

        original_finalize = config_cls.finalize

        def finalize_without_core_dsa_cp_assertion(
            self,
            *args,
            __original_finalize=original_finalize,
            __class_name=class_name,
            **kwargs,
        ):
            _patch_glm_moe_dsa_bridge_config(
                self,
                f"megatron.bridge.models.transformer_config.{__class_name}.finalize",
            )
            _clear_dsa_context_parallel_hint(
                self,
                f"megatron.bridge.models.transformer_config.{__class_name}.finalize",
            )
            return __original_finalize(self, *args, **kwargs)

        config_cls.finalize = finalize_without_core_dsa_cp_assertion
        config_cls._modal_glm_moe_dsa_finalize_patch = True
        _log_patch(f"patched bridge {class_name}.finalize")


def _patch_megatron_core_dsa_context_parallel_guard() -> None:
    try:
        import megatron.core.transformer.transformer_config as transformer_config
    except Exception:
        return

    for class_name in ("TransformerConfig", "MLATransformerConfig"):
        config_cls = getattr(transformer_config, class_name, None)
        if (
            config_cls is None
            or config_cls.__dict__.get("_modal_glm_moe_dsa_cp_patch", False)
        ):
            continue

        original_post_init = config_cls.__post_init__

        def post_init_without_core_dsa_cp_assertion(
            self, __original_post_init=original_post_init, __class_name=class_name
        ):
            # GLM-5 uses Slime's DSAMLASelfAttention replacement, which handles
            # context parallelism itself. Megatron-Core validates its own DSA
            # path before Slime swaps the module, so clear this bridge hint here.
            _clear_dsa_context_parallel_hint(
                self,
                f"megatron.core.transformer.transformer_config.{__class_name}.__post_init__",
            )
            return __original_post_init(self)

        config_cls.__post_init__ = post_init_without_core_dsa_cp_assertion
        config_cls._modal_glm_moe_dsa_cp_patch = True
        _log_patch(f"patched Megatron-Core {class_name}.__post_init__")


def _patch_slime_megatron_actor_init() -> None:
    if not _env_enabled("PATCH_SLIME_MEGATRON_ACTOR_INIT", default=True):
        return

    try:
        import slime.backends.megatron_utils.actor as actor_module
    except Exception:
        return

    actor_cls = actor_module.MegatronTrainRayActor
    if getattr(actor_cls, "_modal_runtime_patches_init_patch", False):
        return

    original_init = actor_cls.init

    def init_with_modal_runtime_patches(self, *args, **kwargs):
        apply_modal_runtime_patches()
        return original_init(self, *args, **kwargs)

    actor_cls.init = init_with_modal_runtime_patches
    actor_cls._modal_runtime_patches_init_patch = True
    _log_patch("patched MegatronTrainRayActor.init")


def _patch_megatron_allgather_cp_rope() -> None:
    if not _env_enabled("PATCH_MEGATRON_ALLGATHER_CP_ROPE"):
        return

    try:
        import torch
        import megatron.core.models.common.embeddings.rope_utils as rope_utils
    except Exception:
        return

    if getattr(rope_utils, "_modal_allgather_cp_rope_patch", False):
        return

    original_thd = rope_utils._apply_rotary_pos_emb_thd
    apply_bshd = rope_utils._apply_rotary_pos_emb_bshd

    def slice_freqs(
        freqs,
        start: int,
        length: int,
        fallback_offset: int,
        original_total: int,
    ):
        if length <= 0:
            return freqs.narrow(0, 0, 0)

        freq_len = int(freqs.size(0))
        if freq_len == original_total and start < freq_len:
            end = min(start + length, freq_len)
            piece = freqs[start:end]
        else:
            end = min(fallback_offset + length, freq_len)
            piece = freqs[fallback_offset:end]

        missing = length - int(piece.size(0))
        if missing <= 0:
            return piece

        # Missing positions are synthetic pad tokens. Reusing the final rotary
        # row keeps the tensor shape valid while masked pad tokens contribute no
        # training loss.
        filler = freqs[-1:].expand(missing, *freqs.shape[1:])
        return torch.cat([piece, filler], dim=0)

    def thd_allgather_cp_rope(
        t,
        cu_seqlens,
        freqs,
        rotary_interleaved=False,
        multi_latent_attention=False,
        mscale=1.0,
        cp_group=None,
        **kwargs,
    ):
        if cp_group is None or cu_seqlens is None:
            return original_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
                **kwargs,
            )

        cp_size = cp_group.size()
        if cp_size <= 1:
            return original_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
                **kwargs,
            )

        local_len = int(t.size(0))
        global_len = local_len * cp_size
        global_pad_limit = int(
            os.environ.get("PATCH_MEGATRON_ALLGATHER_CP_ROPE_MAX_PAD_TOKENS", "8192")
        )

        try:
            boundaries = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        except Exception:
            return original_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
                **kwargs,
            )

        if not boundaries:
            return original_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
                **kwargs,
            )

        original_total = int(cu_seqlens[-1].item())

        if boundaries[-1] < global_len:
            pad = global_len - boundaries[-1]
            if pad > global_pad_limit:
                return original_thd(
                    t,
                    cu_seqlens,
                    freqs,
                    rotary_interleaved=rotary_interleaved,
                    multi_latent_attention=multi_latent_attention,
                    mscale=mscale,
                    cp_group=cp_group,
                    **kwargs,
                )
            boundaries.append(global_len)
        elif boundaries[-1] > global_len:
            return original_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
                **kwargs,
            )

        cp_rank = cp_group.rank()
        local_start = cp_rank * local_len
        local_end = local_start + local_len
        pieces = []

        for seq_start, seq_end in zip(boundaries, boundaries[1:], strict=False):
            if seq_end <= seq_start:
                continue
            overlap_start = max(local_start, seq_start)
            overlap_end = min(local_end, seq_end)
            if overlap_start >= overlap_end:
                continue
            pieces.append(
                slice_freqs(
                    freqs,
                    overlap_start,
                    overlap_end - overlap_start,
                    overlap_start - seq_start,
                    original_total,
                )
            )

        if not pieces:
            return original_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
                **kwargs,
            )

        freqs_packed = torch.cat(pieces, dim=0)
        if int(freqs_packed.size(0)) < local_len:
            missing = local_len - int(freqs_packed.size(0))
            freqs_packed = torch.cat(
                [freqs_packed, freqs[-1:].expand(missing, *freqs.shape[1:])],
                dim=0,
            )
        elif int(freqs_packed.size(0)) > local_len:
            freqs_packed = freqs_packed[:local_len]

        return apply_bshd(
            t.unsqueeze(1),
            freqs_packed,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=multi_latent_attention,
            mscale=mscale,
        ).squeeze(1)

    thd_allgather_cp_rope._modal_allgather_cp_rope_patch = True
    thd_allgather_cp_rope._modal_original = original_thd
    rope_utils._apply_rotary_pos_emb_thd = thd_allgather_cp_rope
    rope_utils._modal_allgather_cp_rope_patch = True
    _log_patch("patched Megatron THD RoPE for allgather-CP layout")


def apply_modal_runtime_patches() -> None:
    """Apply patches selected by environment variables."""
    _patch_fp8_dequant_cuda()
    _patch_flattened_tensor_bucket_single_tensor_zero_copy()
    _patch_dsa_kv_pool_cpu_copy_signature()
    _patch_glm_moe_dsa_bridge_context_parallel()
    _patch_megatron_allgather_cp_rope()
    _patch_slime_megatron_actor_init()
    _patch_dcp_thread_count()
    _patch_dcp_buffered_torch_save()
