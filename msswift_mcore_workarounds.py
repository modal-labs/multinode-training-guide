import os
import random
from argparse import Namespace
from typing import Any

import numpy as np
import torch


def _load_iteration_from_tracker(tracker_path: str) -> int:
    with open(tracker_path, "r", encoding="utf-8") as f:
        return int(f.read().strip())


def _prune_extra_state_entries(obj: Any) -> int:
    removed = 0
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if "_extra_state" in key:
                del obj[key]
                removed += 1
            else:
                removed += _prune_extra_state_entries(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            removed += _prune_extra_state_entries(item)
    return removed


def _strip_prefix_from_nested_keys(obj: Any, prefix: str, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    renamed = 0
    if hasattr(obj, "key") and isinstance(obj.key, str) and obj.key.startswith(prefix):
        obj.key = obj.key[len(prefix) :]
        renamed += 1
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            value = obj.pop(key)
            new_key = key[len(prefix) :] if isinstance(key, str) and key.startswith(prefix) else key
            if new_key != key:
                renamed += 1
            obj[new_key] = value
            renamed += _strip_prefix_from_nested_keys(value, prefix, seen)
    elif isinstance(obj, list):
        for item in obj:
            renamed += _strip_prefix_from_nested_keys(item, prefix, seen)
    elif isinstance(obj, tuple):
        for item in obj:
            renamed += _strip_prefix_from_nested_keys(item, prefix, seen)
    elif hasattr(obj, "__dict__"):
        renamed += _strip_prefix_from_nested_keys(vars(obj), prefix, seen)
    return renamed


def load_mcore_checkpoint_lenient(
    args,
    ddp_models: list,
    optimizer=None,
    opt_param_scheduler=None,
    load_arg: str = "mcore_model",
    adapter_name: str = "default",
):
    from megatron.core import dist_checkpointing, mpu, tensor_parallel
    from swift.megatron import utils as mg_utils
    from swift.megatron.utils import megatron_lm_utils as ml_utils

    if load_arg in {"mcore_adapter", "mcore_ref_adapter"}:
        is_peft_format = True
    else:
        is_peft_format = False

    load_dir = getattr(args, load_arg)
    no_load_optim = args.no_load_optim
    no_load_rng = args.no_load_rng
    finetune = args.finetune
    if not is_peft_format and args.tuner_type != "full":
        no_load_optim = True
        no_load_rng = True
        finetune = False

    models = mg_utils.unwrap_model(ddp_models)
    tracker_path = os.path.join(load_dir, "latest_checkpointed_iteration.txt")
    iteration = _load_iteration_from_tracker(tracker_path)
    checkpoint_dir = os.path.join(load_dir, f"iter_{iteration:07d}")
    state_dict = dist_checkpointing.load_common_state_dict(checkpoint_dir)

    ckpt_tp_pp = (
        state_dict["args"].tensor_model_parallel_size,
        state_dict["args"].pipeline_model_parallel_size,
    )
    run_tp_pp = (args.tensor_model_parallel_size, args.pipeline_model_parallel_size)
    if ckpt_tp_pp != run_tp_pp:
        ml_utils.logger.info(
            f"(TP, PP) mismatch after resume ({run_tp_pp} vs {ckpt_tp_pp} from checkpoint): "
            "RNG state will be ignored"
        )

    gen_sd_rng_state = None
    if (
        ckpt_tp_pp == run_tp_pp
        and not finetune
        and not no_load_rng
        and not getattr(state_dict["args"], "no_save_rng", False)
    ):
        gen_sd_rng_state = mg_utils._get_rng_state()

    sharded_sd_metadata = state_dict.get("content_metadata")
    if (
        not finetune
        and not no_load_optim
        and not getattr(state_dict["args"], "no_save_optim", False)
    ):
        gen_sd_optim = optimizer
        gen_sd_opt_param_scheduler = opt_param_scheduler
        if (
            args.use_distributed_optimizer
            and ckpt_tp_pp != run_tp_pp
            and (sharded_sd_metadata or {}).get("distrib_optim_sharding_type")
            not in {"fully_reshardable", "fully_sharded_model_space", "fsdp_dtensor"}
        ):
            raise RuntimeError("DistributedOptimizer resume is not supported for TP/PP mismatch")
    else:
        gen_sd_optim = None
        gen_sd_opt_param_scheduler = None

    optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
    model_sd_kwargs = dict(metadata=sharded_sd_metadata)
    if is_peft_format:
        sharded_state_dict = {
            "args": Namespace(**vars(args)),
            "checkpoint_version": 3.0,
            "iteration": iteration,
        }
        for i, model in enumerate(models):
            key = "model" if len(models) == 1 else f"model{i}"
            sharded_state_dict[key] = mg_utils.tuners_sharded_state_dict(
                model,
                metadata=sharded_sd_metadata,
            )
    else:
        sharded_state_dict = ml_utils._generate_state_dict(
            args,
            models,
            gen_sd_optim,
            gen_sd_opt_param_scheduler,
            gen_sd_rng_state,
            iteration=iteration,
            model_sd_kwargs=model_sd_kwargs,
            optim_sd_kwargs=optim_sd_kwargs,
        )
    ml_utils._filter_adapter_state_dict(
        sharded_state_dict, is_peft_format, adapter_name=adapter_name
    )
    if is_peft_format:
        renamed = _strip_prefix_from_nested_keys(sharded_state_dict, "base_model.")
        if renamed:
            ml_utils.logger.info(
                f"Stripped 'base_model.' prefix from {renamed} LoRA checkpoint keys before distributed checkpoint load."
            )
    removed = _prune_extra_state_entries(sharded_state_dict)
    if removed:
        ml_utils.logger.info(
            f"Pruned {removed} missing LoRA _extra_state entries before distributed checkpoint load."
        )

    model_keys = [k for k in sharded_state_dict if k.startswith("model")]
    for key in model_keys:
        mg_utils.patch_merge_fn(sharded_state_dict[key])

    load_strategy = ml_utils.get_default_load_sharded_strategy(checkpoint_dir)
    load_strategy = ml_utils.FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )
    state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_dir, load_strategy)

    if finetune:
        iteration = 0
    if "args" in state_dict and not finetune:
        args.consumed_train_samples = getattr(
            state_dict["args"], "consumed_train_samples", 0
        )

    if len(ddp_models) == 1:
        ddp_models[0].load_state_dict(state_dict["model"], strict=False)
    else:
        for i, model in enumerate(ddp_models):
            key = f"model{i}"
            if key in state_dict:
                model.load_state_dict(state_dict[key])

    if not finetune and not no_load_optim:
        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimizer"])
        if opt_param_scheduler is not None:
            opt_param_scheduler.load_state_dict(state_dict["opt_param_scheduler"])
    elif (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()

    if not finetune and not no_load_rng and "rng_state" in state_dict:
        rng_state = state_dict["rng_state"]
        if args.data_parallel_random_init:
            rng_state = rng_state[mpu.get_data_parallel_rank()]
        else:
            rng_state = rng_state[0]
        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    ml_utils.logger.info(f"Successfully loaded Megatron model weights from: {load_dir}")
    return iteration
