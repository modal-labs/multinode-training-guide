import importlib


_LAST_GLM4_HF_CONFIG = None


def _glm4_moe_head_dim(config) -> int:
    qk_nope_head_dim = getattr(config, "qk_nope_head_dim", None)
    if qk_nope_head_dim is not None:
        return qk_nope_head_dim

    v_head_dim = getattr(config, "v_head_dim", None)
    if v_head_dim is not None:
        return v_head_dim

    hidden_size = getattr(config, "hidden_size")
    num_attention_heads = getattr(config, "num_attention_heads")
    return hidden_size // num_attention_heads


def _glm4_moe_rope_theta(config) -> int:
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return rope_theta

    rotary_base = getattr(config, "rotary_base", None)
    if rotary_base is not None:
        return rotary_base

    return 1000000


def _normalize_glm4_moe_config(config) -> None:
    config.rope_theta = _glm4_moe_rope_theta(config)
    config.head_dim = _glm4_moe_head_dim(config)

    if not hasattr(config, "use_qk_norm"):
        config.use_qk_norm = True

    if not hasattr(config, "partial_rotary_factor"):
        config.partial_rotary_factor = 1.0

    if not hasattr(config, "first_k_dense_replace"):
        mlp_layer_types = getattr(config, "mlp_layer_types", None) or []
        first_sparse_idx = next(
            (idx for idx, layer_type in enumerate(mlp_layer_types) if layer_type == "sparse"),
            len(mlp_layer_types),
        )
        config.first_k_dense_replace = first_sparse_idx

    if not hasattr(config, "num_nextn_predict_layers"):
        config.num_nextn_predict_layers = 0

def _register_glm4_moe_lite() -> None:
    try:
        import transformers
        from transformers import AutoConfig
    except Exception:
        return

    try:
        from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM
    except Exception:
        return

    if not isinstance(getattr(Glm4MoeConfig, "head_dim", None), property):
        Glm4MoeConfig.head_dim = property(_glm4_moe_head_dim)

    if not isinstance(getattr(Glm4MoeConfig, "rope_theta", None), property):
        Glm4MoeConfig.rope_theta = property(_glm4_moe_rope_theta)

    try:
        AutoConfig.for_model("glm4_moe_lite")
    except Exception:
        glm4_moe_lite_compat = type(
            "Glm4MoeConfig",
            (Glm4MoeConfig,),
            {
                "model_type": "glm4_moe_lite",
                "head_dim": property(_glm4_moe_head_dim),
                "rope_theta": property(_glm4_moe_rope_theta),
            },
        )
        glm4_moe_lite_compat.__module__ = __name__
        globals()["Glm4MoeConfig"] = glm4_moe_lite_compat
        AutoConfig.register("glm4_moe_lite", glm4_moe_lite_compat, exist_ok=True)

    try:
        getattr(transformers, "Glm4MoeLiteForCausalLM")
    except AttributeError:
        glm4_moe_lite_cls = type(
            "Glm4MoeLiteForCausalLM",
            (Glm4MoeForCausalLM,),
            {},
        )
        glm4_moe_lite_cls.__module__ = (
            "transformers.models.glm4_moe.modeling_glm4_moe"
        )
        globals()["Glm4MoeLiteForCausalLM"] = glm4_moe_lite_cls
        setattr(transformers, "Glm4MoeLiteForCausalLM", glm4_moe_lite_cls)

        lazy_class_map = getattr(transformers, "_class_to_module", None)
        if isinstance(lazy_class_map, dict):
            lazy_class_map["Glm4MoeLiteForCausalLM"] = lazy_class_map.get(
                "Glm4MoeForCausalLM", "models.glm4_moe.modeling_glm4_moe"
            )

        import_structure = getattr(transformers, "_import_structure", None)
        module_name = "models.glm4_moe.modeling_glm4_moe"
        if isinstance(import_structure, dict):
            module_entries = import_structure.setdefault(module_name, [])
            if "Glm4MoeLiteForCausalLM" not in module_entries:
                module_entries.append("Glm4MoeLiteForCausalLM")

        try:
            modeling_module = importlib.import_module(
                "transformers.models.glm4_moe.modeling_glm4_moe"
            )
            setattr(modeling_module, "Glm4MoeLiteForCausalLM", glm4_moe_lite_cls)
        except Exception:
            pass


def _register_megatron_bridge_glm4_moe_lite_alias() -> None:
    try:
        from functools import partial

        import torch
        import transformers
        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron.bridge.models.conversion import model_bridge
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
        from megatron.bridge.models.conversion.peft_bridge import (
            MEGATRON_TO_HF_LORA_SUFFIX,
            MegatronPeftBridge,
        )
        from collections import defaultdict

        from megatron.bridge.models.conversion.param_mapping import (
            AutoMapping,
            ColumnParallelMapping,
            GatedMLPMapping,
            ReplicatedMapping,
            RowParallelMapping,
        )
        from megatron.bridge.models.glm.glm45_bridge import GLM45Bridge
        from megatron.bridge.models.glm.glm45_provider import GLMMoEModelProvider
        from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
        from megatron.bridge.peft.canonical_lora import ModuleDict
    except Exception:
        return

    try:
        glm4_moe = getattr(transformers, "Glm4MoeForCausalLM")
    except AttributeError:
        return

    glm4_moe_lite = getattr(transformers, "Glm4MoeLiteForCausalLM", None)
    if glm4_moe_lite is None:
        glm4_moe_lite = glm4_moe
        setattr(transformers, "Glm4MoeLiteForCausalLM", glm4_moe_lite)

    try:
        import transformer_engine  # noqa: F401

        have_te = True
    except Exception:
        have_te = False

    original_provider_bridge = GLM45Bridge.provider_bridge
    original_build_conversion_tasks = GLM45Bridge.build_conversion_tasks
    original_resolve_hf_adapter_param_name = MegatronPeftBridge._resolve_hf_adapter_param_name
    original_get_base_hf_weight_names_for_adapter = (
        MegatronPeftBridge._get_base_hf_weight_names_for_adapter
    )
    original_build_adapter_conversion_tasks = MegatronPeftBridge.build_adapter_conversion_tasks

    def grouped_expert_hf_weight_names(bridge, global_base_prefix: str, base_suffix: str):
        if not base_suffix and ".weight" in global_base_prefix:
            prefix, weight_suffix = global_base_prefix.rsplit(".weight", 1)
            global_base_prefix = prefix
            base_suffix = f".weight{weight_suffix}"

        if not base_suffix.startswith(".weight"):
            return None

        hf_config = (
            getattr(bridge, "hf_config", None)
            or getattr(bridge, "_hf_config", None)
            or _LAST_GLM4_HF_CONFIG
        )
        hf_num_hidden_layers = getattr(hf_config, "num_hidden_layers", 47)

        if global_base_prefix.startswith("decoder.layers."):
            parts = global_base_prefix.split(".")
            if len(parts) < 4 or not parts[2].isdigit():
                return None
            hf_layer_idx = int(parts[2])
        elif global_base_prefix.startswith("mtp.layers."):
            parts = global_base_prefix.split(".")
            if len(parts) < 4 or not parts[2].isdigit():
                return None
            hf_layer_idx = hf_num_hidden_layers + int(parts[2])
        else:
            return None

        hf_layer_prefix = f"model.layers.{hf_layer_idx}"

        direct_map = {
            ".self_attention.linear_proj": [f"{hf_layer_prefix}.self_attn.o_proj.weight"],
            ".self_attention.linear_q_down_proj": [f"{hf_layer_prefix}.self_attn.q_a_proj.weight"],
            ".self_attention.linear_q_up_proj": [f"{hf_layer_prefix}.self_attn.q_b_proj.weight"],
            ".self_attention.linear_kv_down_proj": [f"{hf_layer_prefix}.self_attn.kv_a_proj_with_mqa.weight"],
            ".self_attention.linear_kv_up_proj": [f"{hf_layer_prefix}.self_attn.kv_b_proj.weight"],
            ".mlp.linear_fc2": [f"{hf_layer_prefix}.mlp.down_proj.weight"],
            ".mlp.shared_experts.linear_fc2": [f"{hf_layer_prefix}.mlp.shared_experts.down_proj.weight"],
        }
        for suffix, names in direct_map.items():
            if global_base_prefix.endswith(suffix):
                return names

        if global_base_prefix.endswith(".mlp.linear_fc1"):
            return [
                f"{hf_layer_prefix}.mlp.gate_proj.weight",
                f"{hf_layer_prefix}.mlp.up_proj.weight",
            ]
        if global_base_prefix.endswith(".mlp.shared_experts.linear_fc1"):
            return [
                f"{hf_layer_prefix}.mlp.shared_experts.gate_proj.weight",
                f"{hf_layer_prefix}.mlp.shared_experts.up_proj.weight",
            ]

        expert_idx_text = base_suffix[len(".weight") :]
        if not expert_idx_text.isdigit():
            return None

        expert_idx = int(expert_idx_text)
        hf_prefix = f"{hf_layer_prefix}.mlp.experts.{expert_idx}"
        if global_base_prefix.endswith(".mlp.experts.linear_fc1"):
            return [
                f"{hf_prefix}.gate_proj.weight",
                f"{hf_prefix}.up_proj.weight",
            ]
        if global_base_prefix.endswith(".mlp.experts.linear_fc2"):
            return [f"{hf_prefix}.down_proj.weight"]
        return None

    def patched_resolve_hf_adapter_param_name(
        self,
        mapping_registry,
        global_base_prefix,
        megatron_suffix,
        base_suffix,
        adapter_key,
    ):
        try:
            return original_resolve_hf_adapter_param_name(
                self,
                mapping_registry,
                global_base_prefix,
                megatron_suffix,
                base_suffix,
                adapter_key,
            )
        except AssertionError as exc:
            if "Expected mapping for adapter base" not in str(exc):
                raise

            fallback_names = grouped_expert_hf_weight_names(self, global_base_prefix, base_suffix)
            if not fallback_names:
                raise

            hf_suffix = MEGATRON_TO_HF_LORA_SUFFIX.get(megatron_suffix)
            if hf_suffix is None:
                raise

            return fallback_names[0][: -len(".weight")] + hf_suffix

    def patched_get_base_hf_weight_names_for_adapter(
        self,
        mapping_registry,
        global_base_prefix,
        adapter_key,
        base_suffix,
    ):
        names = original_get_base_hf_weight_names_for_adapter(
            self,
            mapping_registry,
            global_base_prefix,
            adapter_key,
            base_suffix,
        )
        if names:
            return names

        fallback_names = grouped_expert_hf_weight_names(self, global_base_prefix, base_suffix)
        if fallback_names:
            return fallback_names

        return names

    def patched_build_adapter_conversion_tasks(self, megatron_model):
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        adapters_info = self._megatron_global_adapters_info_all_pp_ranks(megatron_model)
        tasks_by_base = defaultdict(list)
        mapping_registry = self.mapping_registry()

        for (
            global_base_name,
            local_base_prefix,
            input_is_parallel,
            base_linear_is_parallel,
            alpha,
            dim,
            pp_rank,
            vp_stage,
        ) in adapters_info:
            global_base_prefix, _, adapter_suffix = global_base_name.partition(".adapter")
            adapter_key = None
            if adapter_suffix:
                key_token = adapter_suffix.split(".")[-1]
                if key_token.startswith("adapter_"):
                    adapter_key = key_token

            global_linear_in_name, global_linear_out_name = self._construct_adapters_names(
                global_base_prefix, adapter_key
            )
            local_linear_in_name, local_linear_out_name = global_linear_in_name, global_linear_out_name

            base_suffix = ".weight"
            if ".experts." in global_base_prefix and ".local_experts." not in global_base_prefix:
                base_suffix = ".weight0"

            fallback_names = grouped_expert_hf_weight_names(self, global_base_prefix, base_suffix)
            if fallback_names:
                hf_linear_in_name = self._make_lora_param_name(
                    fallback_names[0], ".linear_in.weight"
                )
                hf_linear_out_name = self._make_lora_param_name(
                    fallback_names[0], ".linear_out.weight"
                )
            else:
                hf_linear_in_name = self._resolve_hf_adapter_param_name(
                    mapping_registry,
                    global_base_prefix,
                    ".linear_in.weight",
                    base_suffix,
                    adapter_key,
                )
                hf_linear_out_name = self._resolve_hf_adapter_param_name(
                    mapping_registry,
                    global_base_prefix,
                    ".linear_out.weight",
                    base_suffix,
                    adapter_key,
                )

            linear_in_module, linear_in_weight = None, None
            linear_out_module, linear_out_weight = None, None
            if torch.distributed.is_initialized() and torch.distributed.get_rank() is not None:
                if parallel_state.get_pipeline_model_parallel_rank() == pp_rank:
                    adapter, to_wrap = self._get_adapter_wrap_module(
                        local_base_prefix, megatron_model, vp_stage
                    )
                    if isinstance(adapter, ModuleDict):
                        adapter = adapter[adapter_key]

                    linear_in_module, linear_in_weight = adapter.linear_in, adapter.linear_in.weight
                    linear_out_module, linear_out_weight = adapter.linear_out, adapter.linear_out.weight
                    local_linear_in_name, local_linear_out_name = self._construct_adapters_names(
                        local_base_prefix, adapter_key
                    )

            if base_linear_is_parallel:
                linear_in_mapping_cls = RowParallelMapping if input_is_parallel else ColumnParallelMapping
                linear_out_mapping_cls = ColumnParallelMapping
            else:
                linear_in_mapping_cls = ReplicatedMapping
                linear_out_mapping_cls = ReplicatedMapping

            linear_in_task = WeightConversionTask(
                param_name=local_linear_in_name,
                global_param_name=global_linear_in_name,
                mapping=linear_in_mapping_cls(
                    megatron_param=local_linear_in_name,
                    hf_param=hf_linear_in_name,
                ),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_in_module,
                param_weight=linear_in_weight,
            )
            linear_out_task = WeightConversionTask(
                param_name=local_linear_out_name,
                global_param_name=global_linear_out_name,
                mapping=linear_out_mapping_cls(
                    megatron_param=local_linear_out_name,
                    hf_param=hf_linear_out_name,
                ),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_out_module,
                param_weight=linear_out_weight,
            )
            tasks_by_base[global_base_prefix].append(
                model_bridge.AdapterWeightConversionTask(
                    global_base_prefix=global_base_prefix,
                    adapter_key=adapter_key,
                    alpha=alpha,
                    dim=dim,
                    linear_in_task=linear_in_task,
                    linear_out_task=linear_out_task,
                )
            )

        return tasks_by_base

    class GLM47FlashBridge(GLM45Bridge):
        def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GLMMoEModelProvider:
            global _LAST_GLM4_HF_CONFIG
            hf_config = hf_pretrained.config
            _LAST_GLM4_HF_CONFIG = hf_config
            _normalize_glm4_moe_config(hf_config)

            provider = original_provider_bridge(self, hf_pretrained)

            provider.transformer_layer_spec = partial(
                get_gpt_decoder_block_spec,
                use_transformer_engine=have_te,
            )
            provider.normalization = "RMSNorm"
            provider.gated_linear_unit = True
            provider.add_bias_linear = False
            provider.share_embeddings_and_output_weights = False
            provider.qk_layernorm = True
            provider.multi_latent_attention = True
            provider.q_lora_rank = getattr(hf_config, "q_lora_rank", None)
            provider.kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
            provider.qk_head_dim = getattr(hf_config, "qk_nope_head_dim", provider.kv_channels)
            provider.qk_pos_emb_head_dim = getattr(hf_config, "qk_rope_head_dim", 0)
            provider.v_head_dim = getattr(hf_config, "v_head_dim", provider.kv_channels)
            provider.rope_type = "rope"
            provider.rotary_percent = 1.0
            provider.rotary_scaling_factor = 1.0
            provider.original_max_position_embeddings = getattr(
                hf_config, "max_position_embeddings", provider.seq_length
            )
            provider.beta_fast = 32
            provider.beta_slow = 1
            provider.mscale = 1.0
            provider.mscale_all_dim = 0.0
            provider.cache_mla_latents = False

            provider.moe_shared_expert_overlap = True
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_router_load_balancing_type = "seq_aux_loss"
            provider.moe_router_pre_softmax = True
            provider.moe_grouped_gemm = True
            provider.moe_router_score_function = "sigmoid"
            provider.moe_permute_fusion = True
            provider.moe_router_enable_expert_bias = True
            provider.moe_router_dtype = "fp32"
            provider.moe_router_bias_update_rate = 0
            provider.moe_aux_loss_coeff = 0.0

            provider.apply_rope_fusion = False
            provider.persist_layer_norm = True
            provider.bias_activation_fusion = True
            provider.bias_dropout_fusion = True
            provider.hidden_dropout = 0.0
            provider.autocast_dtype = torch.bfloat16

            provider.mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", None)
            provider.mtp_loss_scaling_factor = 0.3
            provider.moe_shared_expert_intermediate_size = (
                hf_config.moe_intermediate_size * getattr(hf_config, "n_shared_experts", 1)
            )
            provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
                hf_config.num_hidden_layers - hf_config.first_k_dense_replace
            )
            return provider

        def build_conversion_tasks(self, hf_pretrained, megatron_model):
            global _LAST_GLM4_HF_CONFIG
            self._hf_config = hf_pretrained.config
            _LAST_GLM4_HF_CONFIG = hf_pretrained.config
            self._hf_state_source = hf_pretrained.state.source
            self._hf_keys = list(self._hf_state_source.get_all_keys())
            return original_build_conversion_tasks(self, hf_pretrained, megatron_model)

        def mapping_registry(self) -> MegatronMappingRegistry:
            mapping_list = []

            param_mappings = {
                "embedding.word_embeddings.weight": "model.embed_tokens.weight",
                "decoder.final_layernorm.weight": "model.norm.weight",
                "output_layer.weight": "lm_head.weight",
            }

            layer_specific_mappings = {
                "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
                "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
                "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
                "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
                "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
                "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
                "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
                "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
                "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
                "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
                "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
                "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
                "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
                "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
                "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
                "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
                "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            }

            for megatron_param, hf_param in param_mappings.items():
                mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

            for megatron_param, hf_param in layer_specific_mappings.items():
                mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                        gate="model.layers.*.mlp.gate_proj.weight",
                        up="model.layers.*.mlp.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                        gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                        up="model.layers.*.mlp.shared_experts.up_proj.weight",
                    ),
                ]
            )

            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                        up="model.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )

            hf_config = getattr(self, "_hf_config", None)
            if hf_config is None:
                return MegatronMappingRegistry(*mapping_list)

            num_experts = (
                getattr(hf_config, "n_routed_experts", None)
                or getattr(hf_config, "num_local_experts", None)
                or getattr(hf_config, "num_experts", None)
            )
            num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
            num_transformer_layers = hf_config.num_hidden_layers

            def add_exact_grouped_expert_mappings(megatron_prefix: str, hf_prefix: str) -> None:
                if not num_experts:
                    return

                for expert_idx in range(int(num_experts)):
                    mapping_list.extend(
                        [
                            GatedMLPMapping(
                                megatron_param=(
                                    f"{megatron_prefix}.experts.linear_fc1.weight{expert_idx}"
                                ),
                                gate=f"{hf_prefix}.experts.{expert_idx}.gate_proj.weight",
                                up=f"{hf_prefix}.experts.{expert_idx}.up_proj.weight",
                            ),
                            AutoMapping(
                                megatron_param=(
                                    f"{megatron_prefix}.experts.linear_fc2.weight{expert_idx}"
                                ),
                                hf_param=f"{hf_prefix}.experts.{expert_idx}.down_proj.weight",
                            ),
                        ]
                    )

            for layer_idx in range(num_transformer_layers):
                add_exact_grouped_expert_mappings(
                    f"decoder.layers.{layer_idx}.mlp",
                    f"model.layers.{layer_idx}.mlp",
                )

            for mtp_layer in range(num_mtp_layers):
                for layer_prefix in ("mtp_model_layer", "transformer_layer"):
                    for megatron_param, hf_param in layer_specific_mappings.items():
                        megatron_param = (
                            megatron_param.replace(".*", f".*.{layer_prefix}")
                            .replace("decoder", "mtp")
                            .replace(".*", f".{mtp_layer}")
                        )
                        hf_param = hf_param.replace(
                            "layers.*", f"layers.{mtp_layer + num_transformer_layers}"
                        )
                        mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

                    mapping_list.extend(
                        [
                            GatedMLPMapping(
                                megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc1.weight",
                                gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.gate_proj.weight",
                                up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.up_proj.weight",
                            ),
                            GatedMLPMapping(
                                megatron_param=(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.shared_experts.linear_fc1.weight"
                                ),
                                gate=(
                                    f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight"
                                ),
                                up=(
                                    f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight"
                                ),
                            ),
                        ]
                    )

                    mapping_list.extend(
                        [
                            GatedMLPMapping(
                                megatron_param=(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc1.weight*"
                                ),
                                gate=(
                                    f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight"
                                ),
                                up=(
                                    f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight"
                                ),
                            ),
                            AutoMapping(
                                megatron_param=(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc2.weight*"
                                ),
                                hf_param=(
                                    f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.down_proj.weight"
                                ),
                            ),
                        ]
                    )

                    add_exact_grouped_expert_mappings(
                        f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp",
                        f"model.layers.{mtp_layer + num_transformer_layers}.mlp",
                    )

                mapping_list.extend(
                    [
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.enorm.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.hnorm.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.eh_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                            hf_param=(
                                f"model.layers.{mtp_layer + num_transformer_layers}.shared_head.norm.weight"
                            ),
                        ),
                    ]
                )

            return MegatronMappingRegistry(*mapping_list)

        def maybe_modify_converted_hf_weight(
            self,
            task: WeightConversionTask,
            converted_weights_dict,
            hf_state_dict,
        ):
            global_name = task.global_param_name
            if not global_name.startswith("decoder.layers.") or not global_name.endswith(
                ".input_layernorm.weight"
            ):
                return converted_weights_dict

            parts = global_name.split(".")
            if len(parts) < 4 or not parts[2].isdigit():
                return converted_weights_dict

            layer_idx = int(parts[2])
            inv_freq_key = f"model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq"
            if inv_freq_key in converted_weights_dict:
                return converted_weights_dict

            has_inv_freq = getattr(self, "_glm_has_inv_freq", None)
            if has_inv_freq is None:
                has_inv_freq = any(
                    key.startswith("model.layers.") and key.endswith(".self_attn.rotary_emb.inv_freq")
                    for key in hf_state_dict.keys()
                )
                self._glm_has_inv_freq = has_inv_freq
            if not has_inv_freq:
                return converted_weights_dict

            inv_freq = getattr(self, "_glm_inv_freq", None)
            if inv_freq is None:
                rotary_dim = self.hf_config.qk_rope_head_dim
                rotary_base = getattr(self.hf_config, "rope_theta", 10000)
                inv_freq = 1.0 / (
                    rotary_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
                )
                self._glm_inv_freq = inv_freq

            if converted_weights_dict:
                reference_tensor = next(iter(converted_weights_dict.values()))
                if inv_freq.device != reference_tensor.device:
                    inv_freq = inv_freq.to(device=reference_tensor.device)
                    self._glm_inv_freq = inv_freq

            converted_weights_dict[inv_freq_key] = inv_freq
            return converted_weights_dict

    get_bridge = getattr(model_bridge, "get_model_bridge", None)
    registry = getattr(get_bridge, "_exact_types", None)
    if registry is None:
        return

    original_glm_bridge_impl = (
        registry.get(glm4_moe)
        or registry.get(getattr(glm4_moe, "__name__", "Glm4MoeForCausalLM"))
        or registry.get("Glm4MoeForCausalLM")
    )
    if original_glm_bridge_impl is None:
        return

    GLM45Bridge.provider_bridge = GLM47FlashBridge.provider_bridge
    GLM45Bridge.build_conversion_tasks = GLM47FlashBridge.build_conversion_tasks
    GLM45Bridge.mapping_registry = GLM47FlashBridge.mapping_registry
    GLM45Bridge.maybe_modify_converted_hf_weight = (
        GLM47FlashBridge.maybe_modify_converted_hf_weight
    )
    MegatronPeftBridge._resolve_hf_adapter_param_name = patched_resolve_hf_adapter_param_name
    MegatronPeftBridge._get_base_hf_weight_names_for_adapter = (
        patched_get_base_hf_weight_names_for_adapter
    )
    MegatronPeftBridge.build_adapter_conversion_tasks = patched_build_adapter_conversion_tasks
    GLM45Bridge.build_adapter_conversion_tasks = patched_build_adapter_conversion_tasks

    registry[glm4_moe] = original_glm_bridge_impl
    registry[getattr(glm4_moe, "__name__", "Glm4MoeForCausalLM")] = original_glm_bridge_impl
    registry["Glm4MoeForCausalLM"] = original_glm_bridge_impl
    registry[glm4_moe_lite] = original_glm_bridge_impl
    registry[getattr(glm4_moe_lite, "__name__", "Glm4MoeLiteForCausalLM")] = (
        original_glm_bridge_impl
    )
    registry["Glm4MoeLiteForCausalLM"] = original_glm_bridge_impl


def _register_megatron_auto_bridge_glm4_moe_lite_alias() -> None:
    try:
        import transformers
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    except Exception:
        return

    if getattr(AutoBridge, "_glm4_moe_lite_alias_patched", False):
        return

    original_validate_config = AutoBridge._validate_config.__func__
    original_supports = AutoBridge.supports.__func__
    original_causal_lm_architecture = AutoBridge._causal_lm_architecture.func

    def _normalize_architectures(config) -> None:
        architectures = getattr(config, "architectures", None)
        if not architectures:
            return
        normalized = [
            "Glm4MoeForCausalLM" if arch == "Glm4MoeLiteForCausalLM" else arch
            for arch in architectures
        ]
        if normalized != list(architectures):
            config.architectures = normalized

    @classmethod
    def _patched_supports(cls, config):
        _normalize_architectures(config)
        return original_supports(cls, config)

    @classmethod
    def _patched_validate_config(cls, config, path=None):
        _normalize_architectures(config)
        return original_validate_config(cls, config, path)

    def _patched_causal_lm_architecture(self):
        cache_key = "_glm4_moe_lite_causal_lm_architecture"
        if cache_key in self.__dict__:
            return self.__dict__[cache_key]

        architecture = original_causal_lm_architecture(self)
        glm4_moe = getattr(transformers, "Glm4MoeForCausalLM", None)
        glm4_moe_lite = getattr(transformers, "Glm4MoeLiteForCausalLM", None)
        if architecture is glm4_moe_lite and glm4_moe is not None:
            architecture = glm4_moe

        self.__dict__[cache_key] = architecture
        return architecture

    AutoBridge.supports = _patched_supports
    AutoBridge._validate_config = _patched_validate_config
    AutoBridge._causal_lm_architecture = property(_patched_causal_lm_architecture)
    AutoBridge._glm4_moe_lite_alias_patched = True


def _deepseek_ffn_hidden_size(config, layer_idx: int) -> int:
    first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
    if layer_idx < first_k_dense_replace:
        return config.intermediate_size

    moe_hidden_size = getattr(config, "moe_intermediate_size", config.intermediate_size)
    n_shared_experts = getattr(config, "n_shared_experts", 1)
    return moe_hidden_size * n_shared_experts


def _deepseek_get_hidden_dim(self, module_name: str, layer_idx: int):
    config = self.config
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
    qk_nope_head_dim = getattr(config, "qk_nope_head_dim", head_dim - qk_rope_head_dim)
    qk_head_dim = getattr(config, "qk_head_dim", qk_nope_head_dim + qk_rope_head_dim)
    v_head_dim = getattr(config, "v_head_dim", head_dim)
    q_lora_rank = getattr(config, "q_lora_rank", None)
    kv_lora_rank = getattr(config, "kv_lora_rank", None)
    ffn_hidden_size = _deepseek_ffn_hidden_size(config, layer_idx)

    if module_name == "qkv_proj":
        return (
            hidden_size,
            num_attention_heads * qk_head_dim
            + num_key_value_heads * qk_head_dim
            + num_key_value_heads * v_head_dim,
        )
    if module_name == "o_proj":
        return num_attention_heads * v_head_dim, hidden_size
    if module_name == "gate_up_proj":
        return hidden_size, ffn_hidden_size * 2
    if module_name == "down_proj":
        return ffn_hidden_size, hidden_size
    if module_name == "q_b_proj" and q_lora_rank is not None:
        return q_lora_rank, num_attention_heads * qk_head_dim
    if module_name == "kv_b_proj" and kv_lora_rank is not None:
        return kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim)
    if module_name == "q_a_proj" and q_lora_rank is not None:
        return hidden_size, q_lora_rank
    if module_name == "kv_a_proj_with_mqa" and kv_lora_rank is not None:
        return hidden_size, kv_lora_rank + qk_rope_head_dim
    if (
        module_name == "fused_qkv_a_proj_with_mqa"
        and q_lora_rank is not None
        and kv_lora_rank is not None
    ):
        return hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim

    raise NotImplementedError(f"get_hidden_dim not implemented for {module_name}")


def _register_sglang_glm_lora_hidden_dims() -> None:
    candidates = [
        ("sglang.srt.models.deepseek_v2", "DeepseekV2ForCausalLM"),
        ("sglang.srt.models.deepseek_v2", "DeepseekV3ForCausalLM"),
        ("sglang.srt.models.deepseek_v2", "DeepseekV32ForCausalLM"),
        ("sglang.srt.models.glm_moe_dsa", "GlmMoeDsaForCausalLM"),
    ]

    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_cls = getattr(module, class_name)
        except Exception:
            continue

        if getattr(model_cls, "get_hidden_dim", None) is _deepseek_get_hidden_dim:
            continue

        model_cls.get_hidden_dim = _deepseek_get_hidden_dim


def _register_megatron_param_mappings() -> None:
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except Exception:
        return

    # Megatron's fused output layer for vocab-parallel cross entropy is sharded
    # like a column-parallel LM head, but the bridge does not register it.
    AutoMapping.register_module_type("LinearCrossEntropyModule", "column")


def register_transformers_compat() -> None:
    _register_glm4_moe_lite()
    _register_megatron_bridge_glm4_moe_lite_alias()
    _register_megatron_auto_bridge_glm4_moe_lite_alias()
    _register_sglang_glm_lora_hidden_dims()
    _register_megatron_param_mappings()
