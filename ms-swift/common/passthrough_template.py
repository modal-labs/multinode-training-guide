"""Passthrough template plugin for ms-swift Megatron training.

Bypasses ms-swift's template system entirely. Two paths:

  Path A: Raw JSONL — uses HuggingFace's apply_chat_template directly
  Path B: Pre-tokenized (--cached_dataset) — returns input_ids/labels as-is

This avoids three ms-swift template behaviors that alter training data:
  1. _swift_prepare_inputs: merges consecutive same-role messages, reformats tool responses
  2. _add_non_thinking_prefix: prepends </think> to non-thinking assistant responses
  3. _swift_encode: wraps in model-specific prompt tokens differently than HF template

Usage:
    megatron sft --external_plugins /root/passthrough_template.py --template passthrough ...
"""

import json
from typing import Any, Dict

try:
    # ms-swift v4 (reorganized package structure)
    from swift.template import Template, TemplateMeta, register_template
except ImportError:
    # ms-swift v3 (Baseten image)
    from swift.llm import Template, TemplateMeta, register_template


class PassthroughTemplate(Template):
    """Bypass ms-swift template processing entirely.

    Two paths:
      - Pre-tokenized (--cached_dataset): encode() returns input_ids/labels directly,
        skipping TemplateInputs.from_dict() which expects 'messages'.
      - Raw JSONL: _encode() uses HF apply_chat_template for tokenization.

    _preprocess_inputs() is overridden as a no-op to prevent ms-swift's wrap_tool()
    from mangling tool call data before _encode() sees it.
    """

    def encode(self, inputs, **kwargs) -> Dict[str, Any]:
        """Override base encode() to handle pre-tokenized cached datasets.

        ms-swift v4's base encode() calls TemplateInputs.from_dict(inputs) which
        expects a 'messages' key. Pre-tokenized data has input_ids/labels instead.
        Intercept here and return directly.
        """
        if isinstance(inputs, dict) and 'input_ids' in inputs and 'labels' in inputs:
            result = {
                'input_ids': inputs['input_ids'],
                'labels': inputs['labels'],
                'loss_scale': inputs.get('loss_scale'),
            }
            if kwargs.get('return_length'):
                result['length'] = inputs.get('length', len(inputs['input_ids']))
            return result
        return super().encode(inputs, **kwargs)

    def _preprocess_inputs(self, inputs, **kwargs):
        """No-op: bypass ms-swift's tool preprocessing to keep raw data intact."""
        return inputs

    def _encode(self, inputs) -> Dict[str, Any]:
        tokenizer = self.tokenizer

        # Reconstruct full messages list.
        # StdTemplateInputs.from_dict() extracts the system message from messages[0],
        # so inputs.messages won't contain it — we need to put it back.
        messages = list(inputs.messages)
        if inputs.system:
            messages = [{'role': 'system', 'content': inputs.system}] + messages

        # Parse tool_call function.arguments from JSON string to dict,
        # matching preprocess/megatron_*_remote.py behavior via common/preprocessing.preprocess_messages()
        for msg in messages:
            if msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    func = tc.get('function', {})
                    if isinstance(func.get('arguments'), str):
                        try:
                            func['arguments'] = json.loads(func['arguments'])
                        except (json.JSONDecodeError, TypeError):
                            pass

        tools = inputs.tools if inputs.tools else None

        # Find last assistant message (the training target)
        last_assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get('role') == 'assistant':
                last_assistant_idx = idx
                break

        if last_assistant_idx is None:
            raise ValueError(f'No assistant message found in {len(messages)} messages')

        input_messages = messages[:last_assistant_idx]
        full_messages = messages[:last_assistant_idx + 1]

        if not input_messages:
            raise ValueError('No input context (first message is assistant)')

        # Use HF chat template — matches preprocess/megatron_glm_remote.py exactly.
        # enable_thinking=False: prompt ends with </think>, response is non-thinking content.
        formatted_input = tokenizer.apply_chat_template(
            input_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_full = tokenizer.apply_chat_template(
            full_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Output = full conversation minus the input prefix
        if not formatted_full.startswith(formatted_input):
            raise ValueError(
                f'Chat template mismatch: formatted_full does not start with formatted_input. '
                f'Input ends with: ...{formatted_input[-100:]!r}, '
                f'Full starts with: {formatted_full[:100]!r}')
        output_text = formatted_full[len(formatted_input):]

        # Tokenize full sequence together for BPE boundary correctness.
        # Input ends with special tokens that don't BPE-merge across the boundary.
        input_ids = tokenizer.encode(formatted_input + output_text, add_special_tokens=False)
        input_len = len(tokenizer.encode(formatted_input, add_special_tokens=False))
        labels = [-100] * input_len + input_ids[input_len:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'loss_scale': None,
        }


register_template(TemplateMeta(
    template_type='passthrough',
    template_cls=PassthroughTemplate,
    prefix=[],
    prompt=['{{QUERY}}'],
    chat_sep=None,
    suffix=[],
))