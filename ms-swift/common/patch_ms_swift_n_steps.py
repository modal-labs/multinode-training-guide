"""
Patch ms-swift Megatron trainer logging to avoid KeyError on missing `n_steps`.

Some runs produce metric dicts without `n_steps` (e.g. custom hooks / edge cases),
but swift.megatron.trainers.base.on_log currently does `logs.pop("n_steps")`.
This plugin injects a fallback value before delegating to the original method.
"""

import os
from functools import wraps


def _patch_missing_n_steps() -> None:
    try:
        from swift.megatron.trainers import base as trainer_base
    except ImportError:
        return

    target_cls = None
    for cls_name in ("MegatronTrainer", "BaseMegatronTrainer", "Trainer"):
        cls = getattr(trainer_base, cls_name, None)
        if cls is not None and hasattr(cls, "on_log"):
            target_cls = cls
            break

    if target_cls is None:
        return

    original_on_log = target_cls.on_log
    if getattr(original_on_log, "__name__", "") == "_patched_on_log_with_n_steps":
        return

    warn_once = {"done": False}

    @wraps(original_on_log)
    def _patched_on_log_with_n_steps(self, *args, **kwargs):
        logs = kwargs.get("logs")
        args_list = list(args)
        if logs is None and args_list:
            logs = args_list[0]

        if isinstance(logs, dict):
            logs = dict(logs)
            raw_n_steps = logs.get("n_steps")
            if raw_n_steps is None:
                raw_n_steps = (
                    logs.get("global_step")
                    or logs.get("step")
                    or logs.get("iteration")
                    or logs.get("iter")
                    or logs.get("consumed_train_samples")
                )
            try:
                n_steps = int(raw_n_steps)
            except (TypeError, ValueError):
                n_steps = 0

            if n_steps <= 0:
                n_steps = 1
                rank = int(os.environ.get("RANK", "0"))
                if rank == 0 and not warn_once["done"]:
                    warn_once["done"] = True
                    print(
                        "[ms-swift patch] on_log received invalid n_steps; "
                        f"using n_steps=1. log_keys={sorted(logs.keys())}"
                    )

            logs["n_steps"] = n_steps

        if "logs" in kwargs:
            kwargs["logs"] = logs
            return original_on_log(self, *args, **kwargs)

        if args_list:
            args_list[0] = logs
            return original_on_log(self, *tuple(args_list), **kwargs)

        return original_on_log(self, *args, **kwargs)

    target_cls.on_log = _patched_on_log_with_n_steps


_patch_missing_n_steps()
