"""Turn a finished eval's rollout dump into aggregate numbers.

The ``eval/<dataset>`` scalar slime logs is a floor, not a measurement: an
episode whose image never built ships a masked reward-0 sample (see
``agentic_rl.generate._ship_null``) and counts against the mean exactly like a
task the model genuinely failed. This separates the two, so a run reports both
the raw rate and the rate over episodes that actually ran -- the only one worth
comparing across checkpoints, since build health drifts independently of the
model.

It also splits the failures by repo. The eval slices are repo-disjoint, so a
repo that fails on all of its tasks is a broken Dockerfile that will keep
failing, while a repo that fails on some is transient and a retry recovers it.

Runs where the dump lives so prompts and responses never leave the container --
only counts, rates and instance ids are printed.

Defaults to the most recent dump on the checkpoints volume; pass ``EVAL_RUN`` to
pin a run (newest eval within it) and ``EVAL_ROLLOUT`` to pin a step, which is
how you line a checkpoint up against its baseline:

    uv run --no-dev modal run slime/modal_eval_report.py::report

    EVAL_RUN=qwen3.6-27b-snorkel-1-eval500-base-20260730-150725 \
        uv run --no-dev modal run slime/modal_eval_report.py::report
"""

import os

import modal

CHECKPOINTS = "/checkpoints"
DUMPS = f"{CHECKPOINTS}/swe_rollout_dumps"

# Nothing from the training image is needed -- the dump is plain Python objects
# under a torch pickle -- so a CPU wheel off pytorch's own index keeps this a
# ~200 MB pull instead of the multi-GB CUDA build.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", index_url="https://download.pytorch.org/whl/cpu"
)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints")
app = modal.App("snorkel-eval-report")


@app.function(
    image=image,
    volumes={CHECKPOINTS: checkpoints_volume},
    cpu=2.0,
    memory=32 * 1024,
    timeout=30 * 60,
)
def report(
    run: str = os.environ.get("EVAL_RUN", ""),
    rollout: int = int(os.environ.get("EVAL_ROLLOUT", "-1")),
    top: int = 12,
):
    import re
    import statistics
    from collections import Counter, defaultdict
    from pathlib import Path

    import torch

    # ── Locate the dump ───────────────────────────────────────────────────────
    # A training run writes rollout_eval_<n>.pt at every eval interval, so a run
    # alone no longer pins one file: default to its newest eval, or pass
    # EVAL_ROLLOUT to pick the step (e.g. 0 for the pre-training anchor).
    root = Path(DUMPS)
    pattern = f"rollout_eval_{rollout}.pt" if rollout >= 0 else "rollout_eval_*.pt"
    found = sorted((root / run).glob(f"*/{pattern}" if not run else pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no dumps matching {pattern} under {root / run}")
    path = found[-1]
    print(f"[report] {path}")

    # weights_only=True refuses anything but tensors and primitives; the dump is
    # dicts of those, so it should load clean. Fall back only if a field turns out
    # to hold something richer.
    try:
        blob = torch.load(path, weights_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"[report] weights_only load rejected ({type(e).__name__}), retrying unrestricted")
        blob = torch.load(path, weights_only=False)

    samples = blob["samples"]
    print(f"[report] rollout_id={blob.get('rollout_id')}, {len(samples)} samples")

    # ── Samples -> episodes ───────────────────────────────────────────────────
    # A multi-chain episode ships one sample per chain, each holding reward/k, so
    # the episode's reward is the sum over its chains. Siblings are copies of one
    # Sample, hence identical (index, group_index).
    episodes: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        md = s.get("metadata") or {}
        episodes[(md.get("instance_id"), s.get("index"), s.get("group_index"))].append(s)

    multi = sum(1 for chain in episodes.values() if len(chain) > 1)
    print(f"[report] {len(episodes)} episodes, {multi} of them multi-chain")

    # ── Classify ──────────────────────────────────────────────────────────────
    # Buckets are structural and mutually exclusive. episode_exception rides along
    # as an overlay because an episode can raise mid-run and still have produced
    # usable turns.
    def classify(first: dict) -> tuple[str, str]:
        md = first.get("metadata") or {}
        if first.get("status") == "aborted":
            return "aborted", str(md.get("abort_reason") or "?")
        agentic = md.get("agentic") or {}
        if first.get("remove_sample"):
            reason = str(agentic.get("exit_status") or "?")
            # ImageUnusable means empty chains: the sandbox died before the first
            # LLM call, i.e. the image never came up. Any other exit_status means
            # the model ran and every turn was rolled back.
            return ("image_unusable" if reason == "ImageUnusable" else "no_usable_generation"), reason
        return "graded", ""

    rows = []
    for chain in episodes.values():
        first = chain[0]
        md = first.get("metadata") or {}
        agentic = md.get("agentic") or {}
        bucket, reason = classify(first)
        rows.append(
            {
                "instance": md.get("instance_id"),
                "bucket": bucket,
                "reason": reason,
                "reward": sum(float(s.get("reward") or 0.0) for s in chain),
                "solved": bool(agentic.get("is_solved")),
                "turns": agentic.get("turns"),
                "elapsed": agentic.get("elapsed_sec"),
                "timing": agentic.get("timing") or {},
                "errored": agentic.get("error") == "episode_exception",
                "snorkel": md.get("snorkel") or {},
                "n_samples": len(chain),
            }
        )

    n = len(rows)
    buckets = Counter(r["bucket"] for r in rows)
    infra = [r for r in rows if r["bucket"] in ("aborted", "image_unusable")]
    scored = [r for r in rows if r["bucket"] not in ("aborted", "image_unusable")]
    solved = [r for r in rows if r["solved"]]

    print("\n=== outcome ===")
    for name in ("graded", "no_usable_generation", "image_unusable", "aborted"):
        if buckets.get(name):
            print(f"  {name:22s} {buckets[name]:4d}  ({100 * buckets[name] / n:5.1f}%)")
    if errored := sum(1 for r in rows if r["errored"]):
        print(f"  {'(raised mid-episode)':22s} {errored:4d}  overlay on the buckets above")

    reasons = Counter(r["reason"] for r in rows if r["reason"])
    if reasons:
        print("  terminal reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))

    # ── Solve rate ────────────────────────────────────────────────────────────
    sample_mean = statistics.fmean(float(s.get("reward") or 0.0) for s in samples)
    episode_mean = statistics.fmean(r["reward"] for r in rows)
    raw = len(solved) / n
    adjusted = len(solved) / len(scored) if scored else 0.0

    print("\n=== solve rate ===")
    print(f"  mean reward over samples   {sample_mean:.4f}   <- reconcile against the logged eval/ number")
    print(f"  mean reward over episodes  {episode_mean:.4f}")
    print(f"  raw        {len(solved):3d}/{n:3d}  {100 * raw:5.1f}%")
    print(f"  adjusted   {len(solved):3d}/{len(scored):3d}  {100 * adjusted:5.1f}%   (excludes {len(infra)} infra failures)")
    print(f"  delta      {100 * (adjusted - raw):+.1f}pp")

    # ── Breakdowns ────────────────────────────────────────────────────────────
    def breakdown(field: str) -> None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[str(r["snorkel"].get(field, "?"))].append(r)
        if len(groups) == 1 and "?" in groups:
            print(f"\n=== by {field} ===\n  (not stamped on these rows)")
            return
        print(f"\n=== by {field} ===")
        print(f"  {'value':<16} {'n':>4} {'solved':>7} {'raw':>7} {'infra':>6} {'adj':>7}")
        for value, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            got = sum(1 for r in group if r["solved"])
            bad = sum(1 for r in group if r["bucket"] in ("aborted", "image_unusable"))
            ok = len(group) - bad
            adj = f"{100 * got / ok:6.1f}%" if ok else "     --"
            print(f"  {value:<16} {len(group):4d} {got:7d} {100 * got / len(group):6.1f}% {bad:6d} {adj}")

    for field in ("language", "language_bucket", "difficulty", "category"):
        breakdown(field)


@app.function(
    image=image,
    volumes={CHECKPOINTS: checkpoints_volume},
    cpu=2.0,
    memory=32 * 1024,
    timeout=30 * 60,
)
def trend(run: str = os.environ.get("EVAL_RUN", ""), top: int = 12):
    """One line per in-run eval: the learning curve ``report`` shows one point of.

    Same classification as ``report`` (solved over episodes; infra = aborted or
    image_unusable, excluded from the adjusted rate), so the numbers line up.
    """
    from collections import defaultdict
    from pathlib import Path

    import torch

    root = Path(DUMPS)
    if run:
        dumps = sorted((root / run).glob("rollout_eval_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    else:
        runs = sorted({p.parent for p in root.glob("*/rollout_eval_*.pt")}, key=lambda d: d.stat().st_mtime)
        if not runs:
            raise SystemExit(f"no eval dumps under {root}")
        dumps = sorted(runs[-1].glob("rollout_eval_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    print(f"[trend] {dumps[0].parent.name}: {len(dumps)} evals")
    print(f"  {'rollout':>7} {'raw':>7} {'adjusted':>9} {'infra':>6}")

    for path in dumps:
        blob = torch.load(path, weights_only=True)
        episodes: dict[tuple, list[dict]] = defaultdict(list)
        for s in blob["samples"]:
            md = s.get("metadata") or {}
            episodes[(md.get("instance_id"), s.get("index"), s.get("group_index"))].append(s)

        n = solved = infra = 0
        for chain in episodes.values():
            first = chain[0]
            md = first.get("metadata") or {}
            n += 1
            if first.get("status") == "aborted" or (
                first.get("remove_sample") and (md.get("agentic") or {}).get("exit_status") == "ImageUnusable"
            ):
                infra += 1
            elif (md.get("agentic") or {}).get("is_solved"):
                solved += 1
        rollout_id = int(path.stem.rsplit("_", 1)[-1])
        adj = solved / (n - infra) if n > infra else 0.0
        print(f"  {rollout_id:>7} {100 * solved / n:6.1f}% {100 * adj:8.1f}% {infra:6d}")

    # ── Timing ────────────────────────────────────────────────────────────────
    def pct(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    def spread(label: str, values: list[float]) -> None:
        if not values:
            print(f"  {label:<22} (none recorded)")
            return
        print(
            f"  {label:<22} n={len(values):4d}  p50 {pct(values, 0.5):7.1f}s  p90 {pct(values, 0.9):7.1f}s  "
            f"p99 {pct(values, 0.99):7.1f}s  max {max(values):7.1f}s"
        )

    print("\n=== timing ===")
    spread("elapsed", [float(r["elapsed"]) for r in rows if r["elapsed"] is not None])
    # Split elapsed by bucket: a task that cannot build still burns its build
    # budget before giving up, so dead tasks are not cheap ones. This is what
    # sizes the cost of leaving them in a training pool.
    for bucket in sorted({r["bucket"] for r in rows}):
        spread(
            f"  elapsed/{bucket}",
            [float(r["elapsed"]) for r in rows if r["bucket"] == bucket and r["elapsed"] is not None],
        )
    for phase in ("boot", "image", "prep", "agent", "generate", "verifier"):
        spread(phase, [float(v) for r in rows if (v := r["timing"].get(phase)) is not None])

    turns = [int(r["turns"]) for r in rows if r["turns"] is not None]
    if turns:
        print(f"  turns      p50 {pct(turns, 0.5):.0f}  p90 {pct(turns, 0.9):.0f}  max {max(turns)}")

    # ── Ids worth looking at ──────────────────────────────────────────────────
    slowest = sorted((r for r in rows if r["elapsed"] is not None), key=lambda r: -float(r["elapsed"]))[:top]
    print(f"\n=== slowest {len(slowest)} episodes ===")
    for r in slowest:
        print(f"  {float(r['elapsed']):7.0f}s  turns={str(r['turns']):>4}  solved={str(r['solved']):5s}  {r['instance']}")

    if infra:
        print(f"\n=== infra failures ({len(infra)}) ===")
        for r in infra[:top]:
            print(f"  {r['bucket']:15s} {r['reason']:24s} {r['instance']}")
        if len(infra) > top:
            print(f"  ... and {len(infra) - top} more")

        # A repo that fails on every one of its tasks is a broken Dockerfile and
        # stays broken; one that fails on some is transient (scheduling, build
        # timeout) and would come back with a retry. The slice is repo-disjoint,
        # so whole repos land here intact and the split is meaningful.
        by_repo: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            # <dataset>__<owner>_<repo>__<pr> -> <owner>_<repo>
            repo = re.sub(r"__\d+$", "", str(r["instance"] or "")).split("__", 1)[-1]
            by_repo[repo].append(r)

        total_bad = partial_bad = 0
        all_dead, partial = [], []
        for repo, group in by_repo.items():
            bad = [r for r in group if r["bucket"] in ("aborted", "image_unusable")]
            if not bad:
                continue
            if len(bad) == len(group):
                all_dead.append((repo, len(group)))
                total_bad += len(bad)
            else:
                partial.append((repo, len(bad), len(group)))
                partial_bad += len(bad)

        print(f"\n  {len(by_repo)} repos in the slice")
        print(f"  fully dead:  {len(all_dead):3d} repos, {total_bad:3d} tasks  (deterministic build break)")
        print(f"  partial:     {len(partial):3d} repos, {partial_bad:3d} tasks  (transient -- a retry would recover these)")
        for repo, bad, size in sorted(partial, key=lambda t: -t[1])[:top]:
            print(f"    {bad}/{size:<3d} {repo}")
