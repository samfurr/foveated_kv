"""Perplexity comparison: standard vs foveated cache.

Generates Table (ppl) in the paper.

Usage:
  uv run python paper/scripts/perplexity.py
  uv run python paper/scripts/perplexity.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

import sys, os, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from foveated_kv.mlx_foveated import MLXTierConfig
from foveated_kv.mlx_generate import prefill_and_compress


DIVERSE_TEXT = """The history of computing begins with Charles Babbage's Analytical Engine in the 1830s.
Ada Lovelace wrote what is considered the first computer program. Modern computing emerged
during World War II with machines like Colossus and ENIAC. The transistor, invented at Bell
Labs in 1947, revolutionized electronics. Jack Kilby and Robert Noyce independently invented
the integrated circuit. Moore's Law, proposed in 1965, predicted transistor density doubling
every two years. The ARPANET, precursor to the Internet, first connected in 1969.
Tim Berners-Lee invented the World Wide Web at CERN in 1989.

Photosynthesis converts light energy into chemical energy in plants. Chlorophyll absorbs
light primarily in the blue and red wavelengths. The Calvin cycle fixes carbon dioxide into
glucose. Mitochondria perform cellular respiration, converting glucose back to ATP.
DNA replication occurs during the S phase of the cell cycle. RNA polymerase transcribes
DNA into messenger RNA. Ribosomes translate mRNA into proteins using transfer RNA.

The French Revolution began in 1789 with the storming of the Bastille. Napoleon Bonaparte
rose to power and crowned himself Emperor in 1804. The Congress of Vienna in 1815
redrew the map of Europe. The Industrial Revolution transformed manufacturing in Britain.
Steam engines powered factories and railways. The abolition of slavery progressed
throughout the 19th century. Darwin published On the Origin of Species in 1859.

Quantum mechanics describes the behavior of matter at atomic scales. The Heisenberg
uncertainty principle limits simultaneous knowledge of position and momentum. Schrodinger's
equation governs the wave function of quantum systems. Quantum entanglement allows
correlated measurements across distances. Quantum computing uses qubits that can exist
in superposition. Shor's algorithm threatens RSA encryption. Grover's algorithm provides
quadratic speedup for unstructured search.

Cooking involves complex chemical reactions. The Maillard reaction between amino acids
and reducing sugars creates browning and flavor. Caramelization occurs when sugars are
heated above their melting point. Gluten forms when wheat flour proteins hydrate and
are mechanically worked. Fermentation by yeast produces carbon dioxide and ethanol.
Emulsification combines immiscible liquids like oil and water using an emulsifier.
"""


def compute_ppl(model, tokenizer, text, max_ctx, use_foveated=False, cfg=None):
    """Compute perplexity on text up to max_ctx tokens."""
    tokens = mx.array(tokenizer.encode(text)[:max_ctx])
    n = len(tokens)
    if n < 10:
        return float("nan")

    input_ids = tokens.reshape(1, -1)

    if use_foveated and cfg is not None:
        fov_cache, logits, _ = prefill_and_compress(model, input_ids, cfg)
    else:
        cache = make_prompt_cache(model)
        logits = model(input_ids, cache=cache)

    mx.eval(logits)

    # Shift: predict token[i+1] from logits[i]
    shift_logits = logits[0, :-1, :]  # (n-1, vocab)
    shift_labels = tokens[1:]  # (n-1,)

    # Log softmax
    log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)
    mx.eval(log_probs)

    # Gather log probs at label positions
    nll = 0.0
    count = 0
    batch_size = 256
    for start in range(0, len(shift_labels), batch_size):
        end = min(start + batch_size, len(shift_labels))
        batch_lp = log_probs[start:end]
        batch_lb = shift_labels[start:end]
        gathered = batch_lp[mx.arange(end - start), batch_lb]
        mx.eval(gathered)
        nll -= mx.sum(gathered).item()
        count += end - start

    ppl = math.exp(nll / count)
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    print(f"Model: {args.model}")

    cfg = MLXTierConfig(near_pct=0.10)

    print(f"\n{'Context':>8} {'Std PPL':>10} {'Fov PPL':>10} {'Ratio':>8}")
    print("-" * 40)

    for ctx in [1024, 2048, 4096]:
        text = DIVERSE_TEXT * 20  # repeat to fill context
        std_ppl = compute_ppl(model, tokenizer, text, ctx, use_foveated=False)
        fov_ppl = compute_ppl(model, tokenizer, text, ctx, use_foveated=True, cfg=cfg)
        ratio = fov_ppl / std_ppl if std_ppl > 0 else float("nan")
        print(f"{ctx:>8} {std_ppl:>10.2f} {fov_ppl:>10.2f} {ratio:>7.3f}x")


if __name__ == "__main__":
    main()
