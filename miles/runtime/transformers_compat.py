def _glm4_moe_head_dim(config) -> int:
    qk_nope_head_dim = getattr(config, "qk_nope_head_dim", None)
    if qk_nope_head_dim is not None:
        return qk_nope_head_dim

    qk_head_dim = getattr(config, "qk_head_dim", None)
    if qk_head_dim is not None:
        return qk_head_dim

    v_head_dim = getattr(config, "v_head_dim", None)
    if v_head_dim is not None:
        return v_head_dim

    hidden_size = getattr(config, "hidden_size")
    num_attention_heads = getattr(config, "num_attention_heads")
    return hidden_size // num_attention_heads


def _glm4_moe_qk_head_dim(config) -> int:
    qk_head_dim = getattr(config, "qk_head_dim", None)
    if qk_head_dim is not None:
        return qk_head_dim

    qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
    if qk_nope_head_dim or qk_rope_head_dim:
        return qk_nope_head_dim + qk_rope_head_dim

    return _glm4_moe_head_dim(config)


def _register_glm4_moe_lite() -> None:
    try:
        from transformers import AutoConfig
    except Exception:
        return

    try:
        AutoConfig.for_model("glm4_moe_lite")
        return
    except Exception:
        pass

    try:
        from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
    except Exception:
        return

    if not isinstance(getattr(Glm4MoeConfig, "head_dim", None), property):
        Glm4MoeConfig.head_dim = property(_glm4_moe_head_dim)
    if not isinstance(getattr(Glm4MoeConfig, "qk_head_dim", None), property):
        Glm4MoeConfig.qk_head_dim = property(_glm4_moe_qk_head_dim)

    glm4_moe_lite_compat = type(
        "Glm4MoeConfig",
        (Glm4MoeConfig,),
        {
            "model_type": "glm4_moe_lite",
            "head_dim": property(_glm4_moe_head_dim),
            "qk_head_dim": property(_glm4_moe_qk_head_dim),
        },
    )
    glm4_moe_lite_compat.__module__ = __name__
    globals()["Glm4MoeConfig"] = glm4_moe_lite_compat
    AutoConfig.register("glm4_moe_lite", glm4_moe_lite_compat, exist_ok=True)


def _register_megatron_bridge_glm4_moe_lite_alias() -> None:
    try:
        import transformers
        from megatron.bridge.models.conversion import model_bridge
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

    get_bridge = getattr(model_bridge, "get_model_bridge", None)
    registry = getattr(get_bridge, "_exact_types", None)
    if registry is None:
        return

    bridge = (
        registry.get(glm4_moe)
        or registry.get(getattr(glm4_moe, "__name__", None))
        or registry.get("Glm4MoeForCausalLM")
    )
    if bridge is None:
        return

    registry[glm4_moe_lite] = bridge
    registry[getattr(glm4_moe_lite, "__name__", "Glm4MoeLiteForCausalLM")] = bridge
    registry["Glm4MoeLiteForCausalLM"] = bridge


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


def register_transformers_compat() -> None:
    _register_glm4_moe_lite()
    _register_megatron_bridge_glm4_moe_lite_alias()
    _register_sglang_glm_lora_hidden_dims()
