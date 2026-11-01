"""Verify the --long-tokens value that lands the probes.py long gate at ~166,667 prompt tokens.

Replicates data/sources/qwen38-apple-mlx/probes.py gate_long's exact prompt construction
(rng.Random(0), same FILLER, same insert position) and counts tokens with the same tokenizer
the server serves (local snapshot of mlx-community/Qwen3.8-27B-4bit@10c35caa).
Read-only; touches no server state.
"""
import os
import random

FILLER = [
    "The archive ledger lists shipments of copper wire, grain, and lamp oil for the harbour district.",
    "Inspectors noted that the northern warehouse roof leaked during the autumn storms.",
    "A clerk copied the tide tables twice because the first copy smudged in the damp air.",
    "Merchants argued over the price of salt while the ferry waited at the pier.",
    "The lighthouse keeper recorded wind speed, cloud cover, and the number of passing ships.",
]


def build(target):
    rng = random.Random(0)
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
    sentences = [rng.choice(FILLER) for _ in range(target // 20)]
    sentences.insert(len(sentences) // 10, f"IMPORTANT: the vault access code is {code}. Remember it.")
    return " ".join(sentences) + "\n\nWhat is the vault access code mentioned in the text above? Reply with only the code."


from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/ai-work/m4max/run/models/Qwen3.8-27B-4bit-10c35caa"))

points = []
for target in (166667, 191700):
    sents = target // 20
    n = len(tok(build(target))["input_ids"])
    points.append((sents, n))
    print(f"target={target} sentences={sents} tokens={n}", flush=True)

# linear fit tokens = a + b * sentences
(s1, t1), (s2, t2) = points
b = (t2 - t1) / (s2 - s1)
a = t1 - b * s1
want = 166667
solved_sents = round((want - a) / b)
print(f"fit: tokens = {a:.1f} + {b:.4f} * sentences")
print(f"target for ~{want} tokens: --long-tokens {solved_sents * 20} (sentences={solved_sents})")