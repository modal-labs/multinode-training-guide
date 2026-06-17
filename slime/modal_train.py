import asyncio
import os
import shlex
import subprocess
import tempfile

import modal
import modal.experimental

from configs import get_module, _CONFIGS_DIR
from configs.base import HF_CACHE_PATH, DATA_PATH, CHECKPOINTS_PATH, ModalConfig
from modal_helpers.utils import get_checkpoint_conversion_policy

# ── Experiment (client-side only — feeds decorator params) ────────────────────

experiment = os.environ.get("EXPERIMENT_CONFIG", "")
exp_mod = get_module(experiment) if experiment else None
modal_cfg = exp_mod.modal if exp_mod else None
slime_cfg = exp_mod.slime if exp_mod else None

# ── Image ─────────────────────────────────────────────────────────────────────

SLIME_ROOT = "/root/slime"

image = (
    modal.Image.from_registry(
        modal_cfg.docker_image if modal_cfg else ModalConfig.docker_image
    )
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_python_source("modal_helpers", copy=True)
)
if modal_cfg:
    for patch in modal_cfg.patch_files:
        image = image.add_local_file(
            patch, f"/tmp/{os.path.basename(patch)}", copy=True
        )
    if modal_cfg.local_slime:
        image = image.add_local_dir(
            modal_cfg.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )
    if modal_cfg.image_run_commands:
        image = image.run_commands(*modal_cfg.image_run_commands)
    if modal_cfg.image_env:
        image = image.env(modal_cfg.image_env)

with image.imports():
    from ray.job_submission import JobSubmissionClient
    from modal_helpers.utils import (
        build_train_cmd,
        get_modal_cluster_context,
        prepare_slime_config,
        resolve_checkpoint_ref,
        start_ray_head,
    )

# ── Volumes ───────────────────────────────────────────────────────────────────

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)

modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
}

# ── App ──────────────────────────────────────────────────────────────────────

app = modal.App(os.environ.get("MODAL_APP_NAME") or experiment)

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def run_config_hook(experiment: str, hook_name: str, mounted_volumes) -> None:
    """Reload mounted volumes, run a SlimeConfig hook, then commit them."""
    slime_cfg = get_module(experiment).slime
    for volume in mounted_volumes:
        volume.reload()
    getattr(slime_cfg, hook_name)()
    for volume in mounted_volumes:
        volume.commit()


@app.local_entrypoint()
def list_configs():
    """Print all available experiments."""
    _skip = {"base", "__init__"}
    names = sorted(f.stem for f in _CONFIGS_DIR.glob("*.py") if f.stem not in _skip)
    print("Available experiments:")
    for name in names:
        mod = get_module(name)
        nodes = mod.slime.total_nodes()
        gpu = f"{mod.modal.gpu}:{mod.slime.actor_num_gpus_per_node}"
        mode = "async" if mod.slime.async_mode else "sync"
        print(f"  {name:<40} {nodes} node(s) × {gpu}  ({mode})")


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run the experiment's download_model() against the HF cache volume."""
    run_config_hook(experiment, "download_model", (hf_cache_volume,))


@app.function(image=image, cpu=4.0, memory=8 * 1024, timeout=20 * 60)
def inspect_sglang():
    """DEBUG (qwen3.6 collapse): read the RUNNING sglang source for the
    update_weights / memory_saver offload-resume / mamba-state-pool / qwen3 fused
    expert load_weights paths — to pinpoint the sglang-side bug."""
    import os
    import subprocess

    import sglang

    root = os.path.dirname(sglang.__file__)
    print(f"sglang {getattr(sglang, '__version__', '?')} at {root}", flush=True)

    def grep(title, pat, ctx=0, include="*.py", maxlen=3500):
        cmd = ["grep", "-rnE", f"--include={include}"]
        if ctx:
            cmd += [f"-A{ctx}"]
        cmd += [pat, root]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        # strip the long root prefix for readability
        out = out.replace(root + "/", "")
        print(f"\n=========== {title} ===========\n{out[:maxlen]}", flush=True)

    # 1) where update_weights_from_tensor lives + whether it flushes / resets cache
    grep("update_weights_from_tensor defs", r"def update_weights_from_tensor")
    grep("flush_cache / tree_cache reset on weight update", r"flush_cache|self\.tree_cache\.reset|init_next_round|cache\.reset\(\)")
    # 2) mamba / hybrid / deltanet state pool + reset/clear
    grep("mamba/hybrid state pool classes", r"class .*(Mamba|Hybrid|Linear).*(Pool|Cache|StatePool)")
    grep("mamba state pool reset/clear/free", r"(mamba|state).{0,20}(reset|clear|free_all)\(", 1)
    # 3) memory_saver release/resume tags — does offload/resume cover the state pool?
    grep("release/resume_memory_occupation handlers", r"def (release|resume)_memory_occupation", 12)
    # 4) qwen3 fused-expert load_weights (per-expert experts.{N}.gate_proj -> fused)
    grep("qwen3 model files", r"class Qwen3", 0)
    grep("expert_params_mapping / gate_up_proj in qwen3", r"expert_params_mapping|gate_up_proj|make_expert_params", 0)
    # 5) weights_checker endpoint (semantics of check_weights save/compare)
    grep("weights_checker handler", r"weights_checker|def .*weights_check|action == ['\"](save|compare)", 8)
    # 6) qwen3_next load_weights — how experts + deltanet/conv1d are applied on update
    qn = f"{root}/srt/models/qwen3_next.py"
    out = subprocess.run(["grep", "-nE", r"def load_weights|expert|gate_up|conv1d|in_proj|A_log|stacked_params|params_dict", qn], capture_output=True, text=True).stdout
    print(f"\n=========== qwen3_next.py load_weights / expert / deltanet ===========\n{out[:3500]}", flush=True)
    # 7) model_runner update_weights_from_tensor body — does it reset state / reload?
    grep("model_runner.update_weights_from_tensor body", r"def update_weights_from_tensor", 30, maxlen=4000)
    # 8) memory_saver resume — does it cover deltanet/mamba weight buffers?
    grep("memory_saver resume / tags", r"def resume|GPU_MEMORY_TYPE_WEIGHTS|self\.memory_saver_adapter\.resume", 4)
    print("\n[inspect] done", flush=True)


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def post_process_model(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run optional GPU model post-processing for the experiment."""
    run_config_hook(
        experiment,
        "post_process_model",
        (hf_cache_volume, data_volume, checkpoints_volume),
    )


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_data(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run the experiment's download_data() against the data volume."""
    run_config_hook(experiment, "download_data", (data_volume,))


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def post_process_data(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """Run optional GPU data post-processing for the experiment."""
    run_config_hook(
        experiment,
        "post_process_data",
        (hf_cache_volume, data_volume, checkpoints_volume),
    )


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(
        get_checkpoint_conversion_policy(slime_cfg)[0], rdma=True
    )
    if slime_cfg
    else lambda fn: fn
)
def convert_hf_to_megatron_checkpoint(
    experiment: str = os.environ.get("EXPERIMENT_CONFIG", ""),
):
    """Convert HF checkpoint to Megatron torch_dist format in raw mode."""
    slime_cfg = get_module(experiment).slime

    if getattr(slime_cfg, "megatron_to_hf_mode", None) == "bridge":
        print(f"Experiment {experiment!r} is in bridge mode — no conversion needed.")
        return

    hf_cache_volume.reload()
    checkpoints_volume.reload()

    conversion_hf_checkpoint = (
        getattr(slime_cfg, "megatron_conversion_hf_checkpoint", None)
        or slime_cfg.hf_checkpoint
    )
    hf_path = resolve_checkpoint_ref(conversion_hf_checkpoint)
    save_path = str(slime_cfg.ref_load)
    num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(slime_cfg)
    node_rank, master_addr, _, nnodes = get_modal_cluster_context(num_nodes)

    torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
    if nnodes > 1:
        torchrun_args += [
            f"--nnodes={nnodes}",
            f"--node-rank={node_rank}",
            f"--master-addr={master_addr}",
            "--master-port=12355",
        ]

    # For multi-node, use our wrapper that honours SKIP_RELEASE_RENAME to
    # prevent volume corruption (see modal_helpers/convert_hf_to_torch_dist.py).
    # Single-node uses the upstream script directly.
    import importlib.util

    convert_script = (
        importlib.util.find_spec("modal_helpers.convert_hf_to_torch_dist").origin
        if num_nodes > 1
        else f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"
    )

    cmd = (
        f"source {SLIME_ROOT}/{slime_cfg.slime_model_script} && "
        f"torchrun {' '.join(torchrun_args)} {convert_script} "
        f"${{MODEL_ARGS[@]}} {' '.join(extra_args)} "
        f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
    )

    env = {**os.environ, **slime_cfg.environment}
    if num_nodes > 1:
        env["SKIP_RELEASE_RENAME"] = "1"

    print(
        f"Conversion layout for {experiment!r}: nodes={num_nodes}, "
        f"nproc_per_node={nproc_per_node}, node_rank={node_rank}"
    )
    print(
        f"Converting HF checkpoint {conversion_hf_checkpoint!r} "
        f"to Megatron torch_dist at {save_path!r}"
    )
    print(f"Running: bash -c {cmd!r}")
    subprocess.run(["bash", "-c", cmd], check=True, env=env)
    checkpoints_volume.commit()

    if node_rank == 0:
        print(f"Saved Megatron torch_dist checkpoint to {save_path}")


@app.function(
    image=image,
    volumes=modal_volumes,
    cpu=8.0,
    memory=140 * 1024,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def roundtrip_converter_check(
    experiment: str = os.environ.get("EXPERIMENT_CONFIG", ""),
):
    """DEBUG (qwen3.6 first-update collapse): round-trip ref_load (Megatron
    torch_dist) -> HF through the SAME raw converter (`convert_to_hf`) used by
    the live Megatron->SGLang resync, then diff vs the origin HF checkpoint to
    localize which params the conversion corrupts.

    Bisection: corruption here  => bug is in the name/layout mapping
    (megatron_to_hf/qwen3_5.py). Clean here => mapping is fine; the bug is in the
    live TP/EP all-gather (update_weight/common.py), only exercised at TP>1/EP>1
    and NOT by this offline (no_dist) read.
    """
    import json
    import os
    import re
    import sys
    import traceback
    from collections import defaultdict

    import torch
    import torch.distributed.checkpoint as dist_cp
    from safetensors import safe_open

    hf_cache_volume.reload()
    checkpoints_volume.reload()
    slime_cfg = get_module(experiment).slime
    origin_hf = resolve_checkpoint_ref(
        getattr(slime_cfg, "megatron_conversion_hf_checkpoint", None)
        or slime_cfg.hf_checkpoint
    )
    input_dir = f"{slime_cfg.ref_load}/release"
    model_name = "qwen3_5"
    print(f"[roundtrip] origin_hf={origin_hf}", flush=True)
    print(f"[roundtrip] input_dir={input_dir}", flush=True)

    sys.path.insert(0, f"{SLIME_ROOT}/tools")
    sys.path.insert(0, SLIME_ROOT)
    import convert_torch_dist_to_hf as T  # loader + expert/layer unflatten
    from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf

    megatron_args = torch.load(
        os.path.join(input_dir, "common.pt"), weights_only=False
    )["args"]
    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=T.WrappedStorageReader(input_dir),
        planner=T.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"[roundtrip] loaded {len(state_dict)} megatron tensors", flush=True)

    o_idx = json.load(
        open(os.path.join(origin_hf, "model.safetensors.index.json"))
    )["weight_map"]
    o_keys = set(o_idx.keys())
    o_handles: dict = {}

    def o_get(key):
        f = os.path.join(origin_hf, o_idx[key])
        h = o_handles.get(f)
        if h is None:
            h = safe_open(f, framework="pt", device="cpu")
            o_handles[f] = h
        return h.get_tensor(key)

    def norm_to_origin(k):
        if k in o_keys:
            return k
        for cand in (
            k.replace("model.language_model.", "model."),
            k.replace("model.language_model.", ""),
            "model." + k,
        ):
            if cand in o_keys:
                return cand
        return None

    def fam(k):
        k = re.sub(r"\.layers\.\d+\.", ".layers.{L}.", k)
        k = re.sub(r"\.experts\.\d+\.", ".experts.{E}.", k)
        return k

    fam_stats: dict = defaultdict(
        lambda: {
            "n": 0,
            "matched": 0,
            "max": 0.0,
            "sum_mean": 0.0,
            "nan": 0,
            "shape_mismatch": 0,
            "missing": 0,
        }
    )
    worst: list = []
    produced: set = set()
    n_conv = 0
    errors = 0
    for mname, mparam in T.get_named_params(megatron_args, state_dict):
        try:
            converted = convert_to_hf(megatron_args, model_name, mname, mparam)
        except Exception as e:
            errors += 1
            if errors <= 20:
                print(f"[roundtrip] CONVERT ERROR {mname}: {e!r}", flush=True)
            continue
        for hf_name, t in converted:
            n_conv += 1
            produced.add(hf_name)
            f = fam(hf_name)
            s = fam_stats[f]
            s["n"] += 1
            ok = norm_to_origin(hf_name)
            if ok is None:
                s["missing"] += 1
                worst.append((float("inf"), hf_name, "MISSING_IN_ORIGIN"))
                continue
            a = o_get(ok).float()
            b = t.float()
            if a.shape != b.shape:
                s["shape_mismatch"] += 1
                worst.append(
                    (float("inf"), hf_name, f"SHAPE orig{tuple(a.shape)} conv{tuple(b.shape)}")
                )
                continue
            d = (a - b).abs()
            md = d.max().item()
            mn = d.mean().item()
            amax = a.abs().max().item()
            nan = bool(torch.isnan(b).any().item() or torch.isinf(b).any().item())
            s["matched"] += 1
            s["max"] = max(s["max"], md)
            s["sum_mean"] += mn
            if nan:
                s["nan"] += 1
            worst.append(
                (md, hf_name, f"max={md:.3e} mean={mn:.3e} relmax={md/(amax+1e-9):.2e} nan={nan}")
            )

    print(f"[roundtrip] converted {n_conv} hf tensors, convert_errors={errors}", flush=True)
    only_origin = sorted(o_keys - {norm_to_origin(p) for p in produced} - {None})

    worst.sort(key=lambda x: (x[0] == float("inf"), x[0]), reverse=True)
    print("\n===== TOP 40 PARAMS BY MAX ABS DIFF =====", flush=True)
    for md, name, detail in worst[:40]:
        print(f"  {detail:62s}  {name}", flush=True)
    print("\n===== PER-FAMILY SUMMARY (sorted by max abs diff) =====", flush=True)
    for f, s in sorted(fam_stats.items(), key=lambda kv: kv[1]["max"], reverse=True):
        mm = s["sum_mean"] / max(1, s["matched"])
        print(
            f"  max={s['max']:.3e} mean_mean={mm:.3e} matched={s['matched']}/{s['n']} "
            f"shape_mm={s['shape_mismatch']} missing={s['missing']} nan={s['nan']}  {f}",
            flush=True,
        )
    print(f"\n===== ORIGIN KEYS NOT PRODUCED BY CONVERTER ({len(only_origin)}) =====", flush=True)
    for k in only_origin[:30]:
        print(f"  {k}", flush=True)

    report = {
        "families": {k: v for k, v in fam_stats.items()},
        "worst": [(None if md == float("inf") else md, n, dt) for md, n, dt in worst[:300]],
        "only_origin": only_origin[:200],
        "n_conv": n_conv,
        "convert_errors": errors,
    }
    out = f"{CHECKPOINTS_PATH}/_roundtrip_report.json"
    json.dump(report, open(out, "w"), indent=2, default=str)
    checkpoints_volume.commit()
    print(f"\n[roundtrip] wrote {out}", flush=True)


@app.function(
    image=image,
    volumes=modal_volumes,
    cpu=8.0,
    memory=140 * 1024,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def expert_layout_probe(
    experiment: str = os.environ.get("EXPERIMENT_CONFIG", ""),
):
    """DEBUG: the routed-expert grouped path (qwen3_5.py:73-76) passes Megatron
    linear_fc1/linear_fc2 straight to HF fused gate_up_proj/down_proj with NO
    transpose/reorder. Compare the Megatron grouped expert tensors to the origin
    HF fused tensors under candidate transforms to find the exact layout fix.
    """
    import json
    import os
    import re
    import sys

    import torch
    import torch.distributed.checkpoint as dist_cp
    from safetensors import safe_open

    hf_cache_volume.reload()
    checkpoints_volume.reload()
    slime_cfg = get_module(experiment).slime
    origin_hf = resolve_checkpoint_ref(
        getattr(slime_cfg, "megatron_conversion_hf_checkpoint", None)
        or slime_cfg.hf_checkpoint
    )
    input_dir = f"{slime_cfg.ref_load}/release"
    sys.path.insert(0, f"{SLIME_ROOT}/tools")
    sys.path.insert(0, SLIME_ROOT)
    import convert_torch_dist_to_hf as T  # noqa

    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=T.WrappedStorageReader(input_dir),
        planner=T.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    # 1) megatron expert key templates + shapes (layer 0)
    print("\n===== MEGATRON expert tensors (layer 0) =====", flush=True)
    mega = {}
    for k, v in state_dict.items():
        if "expert" in k and re.search(r"\.layers\.0\.", k) and "_extra_state" not in k:
            print(f"  {k}  shape={tuple(v.shape)} dtype={v.dtype}", flush=True)
            mega[k] = v

    # 2) origin fused expert tensors (layer 0)
    o_idx = json.load(
        open(os.path.join(origin_hf, "model.safetensors.index.json"))
    )["weight_map"]

    def o_get(key):
        with safe_open(os.path.join(origin_hf, o_idx[key]), framework="pt", device="cpu") as h:
            return h.get_tensor(key)

    print("\n===== ORIGIN fused expert tensors (layer 0) =====", flush=True)
    pref = "model.language_model.layers.0.mlp.experts"
    o_gu = o_get(f"{pref}.gate_up_proj")
    o_dn = o_get(f"{pref}.down_proj")
    print(f"  {pref}.gate_up_proj shape={tuple(o_gu.shape)} dtype={o_gu.dtype}", flush=True)
    print(f"  {pref}.down_proj    shape={tuple(o_dn.shape)} dtype={o_dn.dtype}", flush=True)

    def find(substr):
        for k, v in mega.items():
            if substr in k:
                return v
        return None

    fc1 = find("linear_fc1.weight")
    fc2 = find("linear_fc2.weight")

    def cmp(name, a, b):
        if a.shape != b.shape:
            print(f"    {name}: SHAPE-MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
            return
        d = (a.float() - b.float()).abs()
        print(f"    {name}: max={d.max():.3e} mean={d.mean():.3e}", flush=True)

    # 3) gate_up: try expert 0 under candidate layouts
    if fc1 is not None and o_gu.dim() == 3:
        E = o_gu.shape[0]
        m = fc1.reshape(E, -1, fc1.shape[-1])[0] if fc1.dim() == 2 else fc1[0]
        o0 = o_gu[0]
        twoI = m.shape[0]
        half = twoI // 2
        print(f"\n[probe] GATE_UP expert0: megatron m={tuple(m.shape)} origin o0={tuple(o0.shape)}", flush=True)
        cmp("m direct", m, o0)
        cmp("m.T", m.transpose(0, 1).contiguous(), o0)
        gate, up = m[:half], m[half:]
        swap = torch.cat([up, gate], dim=0)
        cmp("swap(gate<->up)", swap, o0)
        cmp("swap.T", swap.transpose(0, 1).contiguous(), o0)
        # interleaved gate/up (g0,u0,g1,u1,...)
        inter = torch.stack([gate, up], dim=1).reshape(twoI, -1)
        cmp("interleave", inter, o0)
        cmp("interleave.T", inter.transpose(0, 1).contiguous(), o0)

    # 4) down_proj
    if fc2 is not None and o_dn.dim() == 3:
        E = o_dn.shape[0]
        m2 = fc2.reshape(E, fc2.shape[1] // 1, -1)[0] if fc2.dim() == 2 else fc2[0]
        o2 = o_dn[0]
        print(f"\n[probe] DOWN expert0: megatron m2={tuple(m2.shape)} origin o2={tuple(o2.shape)}", flush=True)
        cmp("m2 direct", m2, o2)
        cmp("m2.T", m2.transpose(0, 1).contiguous(), o2)

    print("\n[probe] done", flush=True)


@app.function(
    image=image,
    volumes=modal_volumes,
    cpu=8.0,
    memory=64 * 1024,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def diff_two_hf(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """DEBUG (qwen3.6 resync probe): diff the faithfully-resynced HF dump
    (/checkpoints/_faithful_resync_hf/iter_0, written by the live TP=2/EP=8 resync
    converter) vs origin HF, per-family. Wrong params => common.py gather/offset bug.
    """
    import json
    import os
    import re
    from collections import defaultdict

    import torch
    from safetensors import safe_open

    hf_cache_volume.reload()
    checkpoints_volume.reload()
    slime_cfg = get_module(experiment).slime
    origin_hf = resolve_checkpoint_ref(
        getattr(slime_cfg, "megatron_conversion_hf_checkpoint", None)
        or slime_cfg.hf_checkpoint
    )
    saved_hf = (
        getattr(slime_cfg, "save_hf", None) or f"{CHECKPOINTS_PATH}/_faithful_resync_hf/iter_{{rollout_id}}"
    ).format(rollout_id=0)
    print(f"[diff] origin={origin_hf}\n[diff] saved={saved_hf}", flush=True)

    def keymap(d):
        """key -> shard file, by scanning every *.safetensors (index-independent)."""
        m = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(d, f), framework="pt", device="cpu") as h:
                    for k in h.keys():
                        m[k] = os.path.join(d, f)
        return m

    o_map, s_map = keymap(origin_hf), keymap(saved_hf)
    print(f"[diff] origin keys={len(o_map)} saved keys={len(s_map)}", flush=True)
    handles: dict = {}

    def get(path, key):
        h = handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt", device="cpu")
            handles[path] = h
        return h.get_tensor(key)

    def fam(k):
        k = re.sub(r"\.layers\.\d+\.", ".layers.{L}.", k)
        k = re.sub(r"\.experts\.\d+\.", ".experts.{E}.", k)
        return k

    fam_stats: dict = defaultdict(
        lambda: {"n": 0, "max": 0.0, "sum_mean": 0.0, "nan": 0, "shape_mm": 0}
    )
    worst: list = []
    common = sorted(set(o_map) & set(s_map))
    for k in common:
        a = get(o_map[k], k).float()
        b = get(s_map[k], k).float()
        f = fam(k)
        s = fam_stats[f]
        s["n"] += 1
        if a.shape != b.shape:
            s["shape_mm"] += 1
            worst.append((float("inf"), k, f"SHAPE orig{tuple(a.shape)} saved{tuple(b.shape)}"))
            continue
        d = (a - b).abs()
        md, mn = d.max().item(), d.mean().item()
        nan = bool(torch.isnan(b).any().item() or torch.isinf(b).any().item())
        s["max"] = max(s["max"], md)
        s["sum_mean"] += mn
        if nan:
            s["nan"] += 1
        worst.append((md, k, f"max={md:.3e} mean={mn:.3e} relmax={md/(a.abs().max().item()+1e-9):.2e} nan={nan}"))

    worst.sort(key=lambda x: (x[0] == float("inf"), x[0]), reverse=True)
    print("\n===== TOP 40 PARAMS BY MAX ABS DIFF (resync vs origin) =====", flush=True)
    for md, name, detail in worst[:40]:
        print(f"  {detail:62s}  {name}", flush=True)
    print("\n===== PER-FAMILY SUMMARY (sorted by max abs diff) =====", flush=True)
    for f, s in sorted(fam_stats.items(), key=lambda kv: kv[1]["max"], reverse=True):
        mm = s["sum_mean"] / max(1, s["n"])
        print(
            f"  max={s['max']:.3e} mean_mean={mm:.3e} n={s['n']} shape_mm={s['shape_mm']} nan={s['nan']}  {f}",
            flush=True,
        )
    print(f"\n[diff] only-in-origin={len(set(o_map) - set(s_map))} only-in-saved={len(set(s_map) - set(o_map))}", flush=True)
    for k in sorted(set(o_map) - set(s_map))[:20]:
        print(f"  only-origin: {k}", flush=True)
    for k in sorted(set(s_map) - set(o_map))[:20]:
        print(f"  only-saved:  {k}", flush=True)
    print("\n[diff] done", flush=True)


@app.function(
    image=image,
    volumes=modal_volumes,
    cpu=8.0,
    memory=96 * 1024,
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def expert_value_check(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    """DEBUG (qwen3.6 resync probe): value-compare the resync's per-expert tensors
    (experts.{N}.gate/up/down_proj) against origin's FUSED experts.gate_up_proj/
    down_proj, per global expert index, to determine whether the resync's expert
    CONTENT+INDEXING is correct (=> pure per-expert-vs-fused format/sglang-ingestion
    bug) or wrong/permuted (=> EP-offset/gather bug in common.py).
    """
    import json
    import os
    from collections import defaultdict

    import torch
    from safetensors import safe_open

    hf_cache_volume.reload()
    checkpoints_volume.reload()
    slime_cfg = get_module(experiment).slime
    origin_hf = resolve_checkpoint_ref(
        getattr(slime_cfg, "megatron_conversion_hf_checkpoint", None) or slime_cfg.hf_checkpoint
    )
    saved_hf = (
        getattr(slime_cfg, "save_hf", None) or f"{CHECKPOINTS_PATH}/_faithful_resync_hf/iter_{{rollout_id}}"
    ).format(rollout_id=0)

    def keymap(d):
        m = {}
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(d, f), framework="pt", device="cpu") as h:
                    for k in h.keys():
                        m[k] = os.path.join(d, f)
        return m

    o_map, s_map = keymap(origin_hf), keymap(saved_hf)
    handles: dict = {}

    def get(mp, key):
        path = mp[key]
        h = handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt", device="cpu")
            handles[path] = h
        return h.get_tensor(key)

    E = int(getattr(slime_cfg, "__dict__", {}).get("num_experts", 0)) or 256
    pre = "model.language_model.layers"
    for L in [0, 20, 39]:
        gu_key = f"{pre}.{L}.mlp.experts.gate_up_proj"
        dn_key = f"{pre}.{L}.mlp.experts.down_proj"
        if gu_key not in o_map:
            print(f"[expert] layer {L}: origin {gu_key} missing", flush=True)
            continue
        gu = get(o_map, gu_key).float()  # [E, 2I, H]
        dn = get(o_map, dn_key).float()  # [E, H, I]
        Eo = gu.shape[0]
        I = gu.shape[1] // 2
        # origin per-expert fingerprints (sum) for permutation detection
        o_fp = {round(gu[e].sum().item(), 2): e for e in range(Eo)}
        aligned = 0
        permuted = 0
        corrupt = 0
        examples = []
        for e in range(Eo):
            gk = f"{pre}.{L}.mlp.experts.{e}.gate_proj.weight"
            uk = f"{pre}.{L}.mlp.experts.{e}.up_proj.weight"
            if gk not in s_map:
                continue
            sg = get(s_map, gk).float()
            su = get(s_map, uk).float()
            # origin expert e: gate = gu[e][:I], up = gu[e][I:]
            dg = (sg - gu[e][:I]).abs().max().item()
            du = (su - gu[e][I:]).abs().max().item()
            if dg == 0 and du == 0:
                aligned += 1
            else:
                # is saved-e actually some other origin index j? (EP permutation)
                fp = round((sg.sum() + su.sum()).item(), 2)
                j = o_fp.get(fp)
                if j is not None and j != e:
                    permuted += 1
                    if len(examples) < 8:
                        examples.append(f"saved e{e} == origin e{j}")
                else:
                    corrupt += 1
                    if len(examples) < 8:
                        examples.append(f"e{e}: max|gate|={dg:.2e} max|up|={du:.2e} (no origin match)")
        print(
            f"[expert] layer {L}: aligned={aligned}/{Eo} permuted={permuted} corrupt={corrupt}  "
            f"examples={examples}",
            flush=True,
        )
    print("\n[expert] done", flush=True)


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}" if modal_cfg else None,
    memory=modal_cfg.memory if modal_cfg and modal_cfg.memory else None,
    cloud=modal_cfg.cloud if modal_cfg and modal_cfg.cloud else None,
    region=modal_cfg.region if modal_cfg and modal_cfg.region else None,
    volumes=modal_volumes,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@(
    modal.experimental.clustered(slime_cfg.total_nodes(), rdma=True)
    if slime_cfg
    else lambda fn: fn
)
async def train(experiment: str = os.environ.get("EXPERIMENT_CONFIG", "")):
    await asyncio.gather(
        hf_cache_volume.reload.aio(),
        data_volume.reload.aio(),
        checkpoints_volume.reload.aio(),
    )
    exp_mod = get_module(experiment)
    slime_cfg = exp_mod.slime
    modal_cfg = exp_mod.modal

    rank, master_addr, my_ip, n_nodes = get_modal_cluster_context(
        slime_cfg.total_nodes()
    )

    os.environ["SLIME_HOST_IP"] = my_ip
    os.environ["SGLANG_HOST_IP"] = my_ip
    os.environ["HOST_IP"] = my_ip

    if rank != 0:
        # Worker node: join the Ray cluster and keep the container alive.
        subprocess.Popen(
            [
                "ray",
                "start",
                f"--node-ip-address={my_ip}",
                "--address",
                f"{master_addr}:{RAY_PORT}",
            ]
        )
        while True:
            await asyncio.sleep(10)

    # Head node: start Ray, prepare config, submit job, stream logs.
    start_ray_head(my_ip, n_nodes)
    prepare_slime_config(slime_cfg, tempfile.mkdtemp())

    if (wandb_key := os.environ.get("WANDB_API_KEY", "")) and getattr(
        slime_cfg, "use_wandb", False
    ):
        slime_cfg.wandb_key = wandb_key

    cmd = build_train_cmd(slime_cfg, SLIME_ROOT)
    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{master_addr}",
            "MASTER_ADDR": master_addr,
            **slime_cfg.environment,
        }
    }

    client = JobSubmissionClient("http://127.0.0.1:8265")
    job_id = client.submit_job(entrypoint=cmd, runtime_env=runtime_env)
    nodes = slime_cfg.total_nodes()
    gpu = f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}"
    mode = "async" if slime_cfg.async_mode else "sync"
    print(f"Job submitted: {job_id}")
    print(f"Training {experiment:<40} {nodes} node(s) × {gpu}  ({mode})")
    print(f"Command: {cmd}, runtime_env: {runtime_env}")

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Ray dashboard: {tunnel.url}")
        async for line in client.tail_job_logs(job_id):
            print(line, end="", flush=True)
