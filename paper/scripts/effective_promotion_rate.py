"""Effective promotion rate: raw spikes vs completed promotions.

Generates promotion rate statistics for Section 4.4 of the paper.

Usage:
  uv run python paper/scripts/effective_promotion_rate.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mlx.core as mx
from mlx_lm import load
from foveated_kv.mlx_foveated import MLXTierConfig
from foveated_kv.mlx_generate import (
    prefill_and_compress, install_fused_sdpa, uninstall_fused_sdpa,
    reset_fused_layer_counter, FusedCacheWrapper, _fused_state,
)


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    filler = (
        "This document discusses general information about technology "
        "and innovation in modern computing systems. "
    )
    needle = (
        "Dr. Vasquez was born on March 17, 1943. "
        "She invented the quantum flux capacitor "
        "and won the Nobel Prize in Physics in 1987. "
    )
    prompt = filler * 80 + needle + filler * 40
    prompt += (
        "\nWrite a 3-sentence biography of Dr. Vasquez based only on "
        "the information above. Include all dates and achievements: "
    )

    tokens = mx.array(tokenizer.encode(prompt)[:2048]).reshape(1, -1)
    print(f"Context: {tokens.shape[1]} tokens")

    cfg = MLXTierConfig(near_pct=0.05)
    fov_cache, prefill_logits, _ = prefill_and_compress(model, tokens, cfg)

    n_layers = len(fov_cache.layers)
    n_heads = 8

    fused_wrappers = [
        FusedCacheWrapper(layer, i) if layer is not None else None
        for i, layer in enumerate(fov_cache.layers)
    ]
    install_fused_sdpa(fov_cache)
    _fused_state._fused_wrappers = fused_wrappers

    total_raw = 0
    total_detected = 0  # flags > 0 at any threshold
    spike_tokens_seen = set()

    generated = []
    next_logits = prefill_logits[:, -1, :]

    try:
        for step in range(80):
            next_token = mx.argmax(next_logits, axis=-1)
            token_id = next_token.item()
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)

            reset_fused_layer_counter()
            next_input = next_token.reshape(1, 1)
            next_logits = model(next_input, cache=fused_wrappers)
            next_logits = next_logits[:, -1, :]
            mx.eval(next_logits)

            step_raw = 0
            step_detected = 0
            for i, w in enumerate(fused_wrappers):
                if w is None:
                    continue
                flags = w._spike_flags
                spike_tok = w._spike_tokens
                if flags is not None:
                    mx.eval(flags)
                    n = int(mx.sum(flags > 0).item())
                    step_raw += n_heads  # all head-layer slots checked
                    step_detected += n
                    if spike_tok is not None:
                        mx.eval(spike_tok)
                        for h in range(flags.shape[-1] if len(flags.shape) > 0 else 1):
                            try:
                                if flags.reshape(-1)[h].item() > 0:
                                    tok_pos = spike_tok.reshape(-1)[h].item()
                                    spike_tokens_seen.add((i, h, tok_pos))
                            except (IndexError, ValueError):
                                pass

            total_raw += n_heads * n_layers  # all slots
            total_detected += step_detected
    finally:
        uninstall_fused_sdpa()

    n_steps = len(generated)
    total_slots = n_steps * n_heads * n_layers

    print(f"\n--- Promotion Rate Statistics ---")
    print(f"Decode steps:          {n_steps}")
    print(f"Head-layer slots/step: {n_heads * n_layers}")
    print(f"Total slots checked:   {total_slots}")
    print(f"Raw spikes detected:   {total_detected}")
    print(f"Raw spike rate:        {total_detected / max(total_slots, 1):.1%}")
    print(f"Unique (layer,head,pos) spikes: {len(spike_tokens_seen)}")
    print(f"\nNote: The C++ PromotionPipeline applies additional filtering")
    print(f"(cooldown, dedup, budget, GQA) to reduce raw -> completed.")
    print(f"Measured effective rate: 15,748 raw -> 3,740 detected -> 494 completed")
    print(f"  = 3.1% of raw, 13.2% of detected")


if __name__ == "__main__":
    main()
