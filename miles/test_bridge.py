"""Quick test: bridge + provider + model construction + mapping validation for K2.5."""
import modal

app = modal.App("test-k25-bridge")
checkpoints_volume = modal.Volume.from_name("miles-checkpoints", create_if_missing=False)

image = (
    modal.Image.from_registry("radixark/miles:dev")
    .entrypoint([])
    .add_local_dir(
        "/home/ec2-user/nan_wonderland/miles",
        remote_path="/root/miles",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
    )
    .add_local_file("patches/megatron_bridge_kimi_vl.patch", "/tmp/megatron_bridge_kimi_vl.patch", copy=True)
    .run_commands(
        "rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/ 2>/dev/null || true",
        "uv pip install --system --no-deps --no-build-isolation git+https://github.com/radixark/Megatron-Bridge.git@d2ee05178d382414bec006fb94dc415483ec6cda",
        "cd $(python -c 'import megatron.bridge; import os; p=megatron.bridge.__file__; print(os.path.dirname(os.path.dirname(os.path.dirname(p))))') && patch -p1 --no-backup-if-mismatch < /tmp/megatron_bridge_kimi_vl.patch",
    )
)

@app.function(image=image, gpu="H200", volumes={"/checkpoints": checkpoints_volume}, timeout=300)
def test():
    import sys
    sys.path.insert(0, "/root/miles")

    # 1. Check patched files for syntax
    import importlib, pathlib
    bridge_dir = pathlib.Path("/usr/local/lib/python3.12/dist-packages/megatron/bridge/models/kimi_vl")
    for py_file in bridge_dir.glob("*.py"):
        try:
            compile(py_file.read_text(), str(py_file), "exec")
            print(f"✓ {py_file.name} syntax OK")
        except SyntaxError as e:
            print(f"✗ {py_file.name} syntax error: {e}")
            return

    # 2. Register bridges
    try:
        import megatron.bridge.models.kimi  # K2Bridge
        import megatron.bridge.models.kimi_vl  # K2.5 VL Bridge
        print("✓ Bridge registration OK")
    except Exception as e:
        print(f"✗ Bridge registration: {e}")
        import traceback; traceback.print_exc()
        return

    # 3. AutoBridge dispatch
    from megatron.bridge import AutoBridge
    try:
        bridge = AutoBridge.from_hf_pretrained("/checkpoints/Kimi-K2.5-bf16", trust_remote_code=True)
        print(f"✓ AutoBridge: {type(bridge._model_bridge).__name__}")
    except Exception as e:
        print(f"✗ AutoBridge: {e}")
        import traceback; traceback.print_exc()
        return

    # 4. Provider
    try:
        provider = bridge.to_megatron_provider(load_weights=False)
        print(f"✓ Provider: {type(provider).__name__}")
        # Check key settings
        print(f"  multi_latent_attention={getattr(provider, 'multi_latent_attention', 'NOT SET')}")
        print(f"  moe_grouped_gemm={getattr(provider, 'moe_grouped_gemm', 'NOT SET')}")
        print(f"  language_only={getattr(provider, 'language_only', 'NOT SET')}")
        print(f"  q_lora_rank={getattr(provider, 'q_lora_rank', 'NOT SET')}")
        print(f"  kv_lora_rank={getattr(provider, 'kv_lora_rank', 'NOT SET')}")
        print(f"  qk_head_dim={getattr(provider, 'qk_head_dim', 'NOT SET')}")
        print(f"  v_head_dim={getattr(provider, 'v_head_dim', 'NOT SET')}")
        print(f"  num_layers={getattr(provider, 'num_layers', 'NOT SET')}")
        print(f"  vocab_size={getattr(provider, 'vocab_size', 'NOT SET')}")
        print(f"  add_bias_linear={getattr(provider, 'add_bias_linear', 'NOT SET')}")
        print(f"  normalization={getattr(provider, 'normalization', 'NOT SET')}")
        print(f"  layernorm_epsilon={getattr(provider, 'layernorm_epsilon', 'NOT SET')}")
    except Exception as e:
        print(f"✗ Provider: {e}")
        import traceback; traceback.print_exc()
        return

    # 5. Check maybe_modify_converted_hf_weight exists and is callable
    try:
        bridge_obj = bridge._model_bridge
        assert hasattr(bridge_obj, 'maybe_modify_converted_hf_weight'), "missing maybe_modify_converted_hf_weight"
        assert hasattr(bridge_obj, '_expert_cache'), "missing _expert_cache"
        assert hasattr(bridge_obj, '_num_moe_experts'), f"missing _num_moe_experts"
        print(f"✓ Expert fusing: _num_moe_experts={bridge_obj._num_moe_experts}, cache initialized")
    except Exception as e:
        print(f"✗ Expert fusing check: {e}")
        import traceback; traceback.print_exc()
        return

    # 5. Test provide() — model construction (requires parallel state init)
    try:
        import os
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=1, rank=0)
        from megatron.core import parallel_state
        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(42)

        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = 1
        provider.expert_tensor_parallel_size = 1
        provider.sequence_parallel = False
        provider.context_parallel_size = 1
        # Build on meta device to avoid OOM (1T model doesn't fit on single GPU)
        import torch as _torch
        provider.params_dtype = _torch.bfloat16
        provider.finalize()
        print(f"✓ Provider finalize OK")
        with _torch.device("meta"):
            model = provider.provide(pre_process=True, post_process=True)
        print(f"✓ Model constructed: {type(model).__name__}")
    except Exception as e:
        print(f"✗ Model construction: {e}")
        import traceback; traceback.print_exc()
        return

    # 6. Validate model param names — check for MLA and GroupedMLP
    param_names = [n for n, _ in model.named_parameters()]
    sample_moe_layer = [n for n in param_names if "layers.4." in n][:10]
    print(f"\n  MoE layer 4 param names (sample):")
    for n in sample_moe_layer:
        print(f"    {n}")

    has_linear_qkv = any("linear_qkv" in n for n in param_names)
    has_linear_q_down = any("linear_q_down_proj" in n for n in param_names)
    has_local_experts = any("local_experts" in n for n in param_names)
    has_grouped_experts = any(".experts.linear_fc" in n for n in param_names)
    print(f"\n  Attention type: {'linear_qkv (WRONG - not MLA!)' if has_linear_qkv else 'MLA projections (correct)'}")
    print(f"  MLA q_down_proj present: {has_linear_q_down}")
    print(f"  Expert type: {'SequentialMLP (local_experts)' if has_local_experts else 'GroupedMLP (fused)'}")
    print(f"  GroupedMLP linear_fc present: {has_grouped_experts}")

    if has_linear_qkv:
        print("\n  ERROR: Model has linear_qkv instead of MLA projections!")
        print("  This means multi_latent_attention is not taking effect.")
        return

    # 7. Validate mapping against model params and checkpoint keys
    import json
    with open("/checkpoints/Kimi-K2.5-bf16/model.safetensors.index.json") as f:
        hf_keys = set(json.load(f)["weight_map"].keys())
    model_params = set(param_names)

    print(f"\n  Model params: {len(model_params)}, HF checkpoint keys: {len(hf_keys)}")

    # Get conversion tasks from bridge
    try:
        tasks = bridge._model_bridge.get_conversion_tasks()
        print(f"  Conversion tasks: {len(tasks)}")

        mapped_megatron = set()
        mapped_hf = set()
        none_tasks = 0
        for t in tasks:
            if t is None:
                none_tasks += 1
                continue
            if hasattr(t, 'megatron_param') and t.megatron_param:
                mapped_megatron.add(t.megatron_param)
            if hasattr(t, 'hf_param') and t.hf_param:
                mapped_hf.add(t.hf_param)

        print(f"  None tasks: {none_tasks}")
        print(f"  Mapped megatron params: {len(mapped_megatron)}")
        print(f"  Mapped HF params: {len(mapped_hf)}")

        # Show sample tasks
        sample = [(getattr(t, 'megatron_param', '?'), getattr(t, 'hf_param', '?'))
                   for t in tasks[:8] if t is not None]
        print(f"\n  Sample mappings:")
        for m, h in sample:
            print(f"    Megatron: {m}  <->  HF: {h}")
    except Exception as e:
        print(f"  Could not get conversion tasks: {e}")

    print("\n  Test complete!")

@app.local_entrypoint()
def main():
    test.remote()
