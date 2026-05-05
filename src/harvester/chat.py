"""Build chat-formatted inputs for prefill.

Olmo-3-Think chat template adds an `<|im_start|>assistant\\n<think>` prefix when
`add_generation_prompt=True`, then the model's sampled tokens follow. The
prefill input we want is therefore:

    apply_chat_template([{user: prompt}], add_generation_prompt=True) + completion_text

We do not append an explicit EOS — the completion may have been truncated at
max_tokens during the original generation, and the captured activations should
correspond exactly to the tokens that were actually rolled out.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def build_input_text(tokenizer: PreTrainedTokenizerBase, prompt: str, completion: str) -> str:
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prefix + completion


def tokenize_input(tokenizer: PreTrainedTokenizerBase, prompt: str, completion: str) -> list[int]:
    text = build_input_text(tokenizer, prompt, completion)
    # add_special_tokens=False because the chat template already includes any BOS/system tokens.
    return tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
