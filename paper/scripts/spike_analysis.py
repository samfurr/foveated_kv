"""Spike detection analysis: frequency, distribution, and layer patterns.

Generates spike statistics for Section 4.4 of the paper.

Usage:
  uv run python paper/scripts/spike_analysis.py
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
    layer0 = fov_cache.layers[0]
    print(f"Layers: {n_layers}")
    print(f"Near valid: {int(mx.max(layer0.near_valid).item())}, "
          f"Far: {layer0.far_k.shape[2]}")

    fused_wrappers = [
        FusedCacheWrapper(layer, i) if layer is not None else None
        for i, layer in enumerate(fov_cache.layers)
    ]
    install_fused_sdpa(fov_cache)
    _fused_state._fused_wrappers = fused_wrappers

    total_spikes = 0
    spike_by_step = []
    spike_by_layer = [0] * n_layers

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

            step_spikes = 0
            for i, w in enumerate(fused_wrappers):
                if w is None:
                    continue
                flags = w._spike_flags
                if flags is not None:
                    mx.eval(flags)
                    n = int(mx.sum(flags > 0).item())
                    step_spikes += n
                    spike_by_layer[i] += n

            total_spikes += step_spikes
            spike_by_step.append(step_spikes)
    finally:
        uninstall_fused_sdpa()

    text = tokenizer.decode(generated)
    n_steps = len(spike_by_step)
    n_heads = 8  # 0.5B model

    print(f"\nGenerated {len(generated)} tokens")
    print(f"\n--- Spike Statistics ---")
    print(f"Total raw spikes:       {total_spikes}")
    print(f"Steps with spikes:      {sum(1 for s in spike_by_step if s > 0)}/{n_steps}")
    print(f"Avg spikes/step:        {total_spikes / max(n_steps, 1):.1f}")
    print(f"Max spikes in one step: {max(spike_by_step) if spike_by_step else 0}")
    print(f"Head-layer slots/step:  {n_heads * n_layers}")
    print(f"Spike rate:             {total_spikes / max(n_steps * n_heads * n_layers, 1):.1%}")

    print(f"\nLayer distribution (all layers):")
    for idx, count in sorted(enumerate(spike_by_layer), key=lambda x: -x[1]):
        if count > 0:
            bar = "#" * min(count // 10, 50)
            print(f"  Layer {idx:>2}: {count:>4} "
                  f"({100 * count / max(total_spikes, 1):.1f}%) {bar}")


if __name__ == "__main__":
    main()
