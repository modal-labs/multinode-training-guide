"""VLM multi-turn rollout for Windows computer use.

Adapted from slime/examples/geo3k_vlm_multi_turn/rollout.py.
This is the custom generate function that Slime calls for each sample.
It manages the interaction loop: model output → parse action →
execute on Windows VM → take screenshot → feed back to model.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

DEFAULT_ENV_MODULE = "custom.windows_computer_use.env_windows"

DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(
            f"rollout_env_{module_path.stem}", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError(
            "Environment module must expose a callable `build_env(sample, args)`."
        )
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        return build_fn(sample, args)


async def _build_env_async(env_module, sample: Sample, args: Any):
    """Build env in a thread to avoid blocking the async event loop."""
    import asyncio as _aio

    return await _aio.to_thread(_build_env, env_module, sample, args)


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}
    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None

    if processor:
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v
            for k, v in processor_output.items()
            if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [
            encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]
        ]

    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    if not chunks:
        return None
    values_by_key: dict = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)
    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)
    return merged


async def _initialize_resources(args: Any, sample: Sample):
    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None))
    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        max_turns = 10

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = await _build_env_async(env_module, sample, args)
    config = {"max_turns": max_turns}
    return env, env_module, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    if processor:
        processor_output = processor(
            text=sample.prompt, **(sample.multimodal_inputs or {})
        )
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v
            for k, v in processor_output.items()
            if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [
            encode_image_for_rollout_engine(img)
            for img in sample.multimodal_inputs["images"]
        ]

    return prompt_ids, image_data, sample.multimodal_train_inputs


def _prepare_start_state(sample, state, args, sampling_params):
    prompt_ids, image_data, init_mm_train = _prepare_initial_inputs(
        sample, state.processor, state.tokenizer
    )
    current_image_data = image_data
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)

    response_tokens: list[int] = (
        sample.tokens[len(prompt_ids) :]
        if len(sample.tokens) >= len(prompt_ids)
        else []
    )
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if getattr(args, "rollout_max_context_len", None) is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)

    return current_image_data, response_tokens, budget, multimodal_train_inputs_buffer


_checked_router = False


async def _run_inference_step(url, tokens, sampling_params, image_data, tokenizer):
    global _checked_router
    if not _checked_router:
        _checked_router = True
        try:
            import aiohttp
            router_base = url.rsplit("/generate", 1)[0]
            async with aiohttp.ClientSession() as sess:
                async with sess.get(f"{router_base}/workers", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    workers = await resp.json()
                    print(f"[inference] Router workers: {workers}")
        except Exception as e:
            print(f"[inference] Failed to query router workers: {e}")

    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    print(f"[inference] POST to {url}: {len(tokens)} tokens, {len(image_data)} images, max_new_tokens={sampling_params.get('max_new_tokens')}")
    import asyncio as _asyncio
    try:
        output = await _asyncio.wait_for(post(url, payload), timeout=300)
    except _asyncio.TimeoutError:
        print(f"[inference] TIMEOUT after 300s!")
        raise
    response_text = output["text"]

    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [
            item[0] for item in output["meta_info"]["output_token_logprobs"]
        ]
    else:
        new_tokens, new_log_probs = [], []

    finish_type = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type


async def _process_env_step(env, response_text, tokenizer, processor, args, sample_metadata):
    import asyncio as _aio

    observation, done, step_info = await _aio.to_thread(env.step, response_text)
    if done:
        return None, None, None, None, True, step_info

    next_user_message = env.format_observation(observation)
    (
        obs_prompt_ids,
        obs_image_data,
        obs_multimodal_inputs,
        obs_multimodal_train_inputs,
    ) = _encode_observation_for_generation(
        tokenizer,
        processor,
        next_user_message,
        sample_metadata,
        getattr(args, "apply_chat_template", True),
        getattr(args, "apply_chat_template_kwargs", None),
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return (
        obs_prompt_ids,
        obs_image_data,
        obs_multimodal_inputs,
        obs_multimodal_train_inputs,
        False,
        step_info,
    )


def _append_to_sample(sample, response_tokens, tokens_to_add, logprobs, loss_mask_val):
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    sample,
    current_image_data,
    obs_image_data,
    obs_multimodal_inputs,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer,
):
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(
            obs_multimodal_inputs, dict
        ):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _should_stop_on_finish(sample, finish_type):
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return True
        case "abort":
            sample.status = Sample.Status.ABORTED
            return True
    return False


def _update_budget(budget, consumed):
    if budget is None:
        return None
    return budget - consumed


def _finalize_sample(
    sample, tokenizer, response_tokens, multimodal_train_inputs_buffer
):
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(
        multimodal_train_inputs_buffer
    )
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


# --------------------------------------------------------------------------
# Main generate function — entry point called by Slime
# --------------------------------------------------------------------------


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn VLM rollout for Windows computer use.

    This is the `--custom-generate-function-path` entry point.
    """
    import time as _time

    _t0 = _time.time()
    print(f"[generate] Starting rollout for sample...")

    assert not getattr(args, "partial_rollout", False), (
        "Partial rollout is not supported for interaction rollouts."
    )

    env, env_module, config, state, url = await _initialize_resources(args, sample)
    print(f"[generate] Env built in {_time.time()-_t0:.0f}s")
    sampling_params = sampling_params.copy()

    current_image_data, response_tokens, budget, multimodal_train_inputs_buffer = (
        _prepare_start_state(sample, state, args, sampling_params)
    )

    all_response_texts: list[str] = []
    try:
        import asyncio as _aio_gen

        print(f"[generate] Resetting env...")
        obs, _ = await _aio_gen.to_thread(env.reset)
        print(f"[generate] Env reset done at {_time.time()-_t0:.0f}s")

        # Encode the initial observation (screenshot) and prepend to context
        if obs.get("multi_modal_data"):
            print(f"[generate] Encoding initial observation...")
            initial_msg = env.format_observation(obs)
            obs_ids, obs_imgs, obs_mm, obs_mm_train = (
                _encode_observation_for_generation(
                    state.tokenizer,
                    state.processor,
                    initial_msg,
                    sample.metadata,
                    getattr(args, "apply_chat_template", True),
                    getattr(args, "apply_chat_template_kwargs", None),
                )
            )
            bos_id = state.tokenizer.bos_token_id
            if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
                obs_ids = obs_ids[1:]

            print(f"[generate] Initial obs encoded: {len(obs_ids)} tokens, budget before={budget}")
            _append_to_sample(
                sample, response_tokens, obs_ids, [0.0] * len(obs_ids), loss_mask_val=0
            )
            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_imgs,
                obs_mm,
                obs_mm_train,
                multimodal_train_inputs_buffer,
            )
            budget = _update_budget(budget, len(obs_ids))
            print(f"[generate] Budget after initial obs: {budget}")

        if budget is not None and budget <= 0:
            print(f"[generate] BUDGET EXHAUSTED after initial obs, returning TRUNCATED")
            sample.status = Sample.Status.TRUNCATED
            return sample

        print(f"[generate] Starting turns (max={config['max_turns']})")
        for turn_idx in range(config["max_turns"]):
            print(f"[generate] Turn {turn_idx+1}/{config['max_turns']} at {_time.time()-_t0:.0f}s")
            cur_sampling_params = sampling_params.copy()
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget

            (
                response_text,
                new_tokens,
                new_logprobs,
                finish_type,
            ) = await _run_inference_step(
                url,
                sample.tokens,
                cur_sampling_params,
                current_image_data,
                state.tokenizer,
            )

            print(f"[generate] Turn {turn_idx+1} inference done: {len(new_tokens)} tokens, finish={finish_type}, text={response_text[:200]!r}")
            all_response_texts.append(response_text)
            _append_to_sample(
                sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1
            )
            budget = _update_budget(budget, len(new_tokens))

            if _should_stop_on_finish(sample, finish_type):
                print(f"[generate] Stopped: finish_type={finish_type}, budget={budget}")
                reward = await _aio_gen.to_thread(env._compute_reward)
                sample.metadata["env_reward"] = reward
                print(f"[generate] End-of-generation reward={reward:.2f}")
                break

            if budget is not None and budget <= 0:
                print(f"[generate] BUDGET EXHAUSTED after turn {turn_idx+1}")
                reward = await _aio_gen.to_thread(env._compute_reward)
                sample.metadata["env_reward"] = reward
                sample.status = Sample.Status.TRUNCATED
                print(f"[generate] Truncated reward={reward:.2f}")
                break

            obs_ids, obs_imgs, obs_mm, obs_mm_train, done, step_info = (
                await _process_env_step(
                    env,
                    response_text,
                    state.tokenizer,
                    state.processor,
                    args,
                    sample.metadata,
                )
            )

            if done:
                # Store the reward in metadata for the RM to read
                reward = (step_info or {}).get("reward", 0.0)
                sample.metadata["env_reward"] = reward
                sample.status = Sample.Status.COMPLETED
                print(f"[generate] Done at turn {turn_idx+1}, reward={reward:.2f}, total={_time.time()-_t0:.0f}s")
                break

            obs_logprobs = [0.0] * len(obs_ids)
            _append_to_sample(
                sample, response_tokens, obs_ids, obs_logprobs, loss_mask_val=0
            )
            budget = _update_budget(budget, len(obs_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_imgs,
                obs_mm,
                obs_mm_train,
                multimodal_train_inputs_buffer,
            )

            if budget is not None and budget <= 0:
                reward = await _aio_gen.to_thread(env._compute_reward)
                sample.metadata["env_reward"] = reward
                sample.status = Sample.Status.TRUNCATED
                print(f"[generate] Truncated (post-obs) reward={reward:.2f}")
                break

            if turn_idx + 1 >= config["max_turns"]:
                # Max turns reached without <done/>, compute reward anyway
                reward = await _aio_gen.to_thread(env._compute_reward)
                sample.metadata["env_reward"] = reward
                sample.status = Sample.Status.COMPLETED
                break

        sample.metadata["all_response_texts"] = all_response_texts
        return _finalize_sample(
            sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer
        )

    except Exception as _exc:
        print(f"[generate] EXCEPTION: {type(_exc).__name__}: {_exc}")
        import traceback
        traceback.print_exc()
        sample.metadata["env_reward"] = 0.0
        sample.metadata["all_response_texts"] = all_response_texts
        sample.status = Sample.Status.COMPLETED
        return _finalize_sample(
            sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer
        )
    finally:
        try:
            await _aio_gen.to_thread(env.close)
        except Exception:
            pass
