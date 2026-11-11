#!/usr/bin/env python3
"""How far the quantized MLX weights are from the unquantized ones: token-wise KL divergence over a sealed panel.

The method of the kld-eval skill, on MLX: one panel of W windows x C tokens, tokenized once and frozen; every model
is run over it (a forward pass, teacher-forced, no generation); at every `stride`-th position the full-vocabulary
next-token distribution of the candidate is held against the teacher's:

    KL = sum over the vocabulary of p_teacher * (log p_teacher - log p_candidate), in nats, summed in float64

Reported: the mean with a window-block bootstrap 95% interval, the median and 95th percentile over positions, and how
often both put the same token first. A control (the candidate's positions shuffled) shows the pairing is not off by one.
Teacher and candidates are loaded together and compared window by window, so no log-prob arrays are kept.

Usage: kld.py build <tokenizer.json> <out panel.npy> <text files...>
       kld.py run <panel.npy> <teacher dir> <out.json> <name=candidate dir>...
"""
import hashlib, json, sys, time
from pathlib import Path

import numpy as np

W, C, STRIDE = 25, 2048, 2


def build(tokenizer_file, out, files):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_file)
    ids = []
    for f in files:  # each file gives whole windows, so no window mixes two kinds of text
        got = tok.encode(Path(f).read_text(encoding="utf-8", errors="ignore"), add_special_tokens=False).ids
        ids.append(got[: len(got) // C * C])
    per = [np.array(x, dtype=np.int32).reshape(-1, C) for x in ids if len(x) >= C]
    order = [w for k in range(max(len(p) for p in per)) for p in per for w in ([p[k]] if k < len(p) else [])]  # round-robin over the files
    panel = np.stack(order[:W])
    assert panel.shape == (W, C), f"the texts give {len(order)} windows of {C} tokens; {W} are needed"
    np.save(out, panel)
    print(json.dumps({"panel": out, "windows": W, "tokens": C, "sha256": hashlib.sha256(panel.tobytes()).hexdigest(), "files": [Path(f).name for f in files]}))


def run(panel_file, teacher_dir, out, candidates):
    import mlx.core as mx
    from mlx_vlm.utils import load

    panel = np.load(panel_file)
    models = {"teacher": load(teacher_dir)[0], **{name: load(path)[0] for name, path in candidates.items()}}

    def logprobs(model, window):  # float64 log-probs at the strided positions of one window
        logits = model(mx.array(window[None])).logits[0, ::STRIDE].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lp)
        return np.array(lp, dtype=np.float64)

    per = {name: [] for name in candidates}  # per window, the KL at each position
    agree = {name: [] for name in candidates}
    control = {name: [] for name in candidates}
    started = time.time()
    for i, window in enumerate(panel):
        t = logprobs(models["teacher"], window)
        p = np.exp(t)
        for name in candidates:
            c = logprobs(models[name], window)
            per[name].append((p * (t - c)).sum(axis=-1))
            agree[name].append((t.argmax(axis=-1) == c.argmax(axis=-1)).mean())
            if i == 0:  # the control: the same candidate with its positions out of step
                control[name] = float((p * (t - np.roll(c, 7, axis=0))).sum(axis=-1).mean())
        print(f"window {i + 1}/{len(panel)}  " + "  ".join(f"{n} {np.mean(per[n][-1]):.5f}" for n in candidates), flush=True)
    rng = np.random.default_rng(20261111)
    result = {"method": "token-wise KL(teacher || candidate), full vocabulary, nats, float64; teacher-forced forward passes",
              "panel": {"file": Path(panel_file).name, "windows": int(panel.shape[0]), "tokens": int(panel.shape[1]), "stride": STRIDE,
                        "positions": int(panel.shape[0] * (panel.shape[1] // STRIDE)), "sha256": hashlib.sha256(panel.tobytes()).hexdigest()},
              "teacher": teacher_dir, "seconds": round(time.time() - started, 1), "candidates": {}}
    for name in candidates:
        windows = np.stack(per[name])  # [W, positions]
        means = windows.mean(axis=1)
        boot = [means[rng.integers(0, len(means), len(means))].mean() for _ in range(10000)]  # whole windows are resampled
        flat = windows.reshape(-1)
        result["candidates"][name] = {"path": candidates[name], "value": float(flat.mean()), "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
                                      "median_tokenwise": float(np.median(flat)), "p95_tokenwise": float(np.percentile(flat, 95)),
                                      "top1_next_token_agreement": float(np.mean(agree[name])), "control_shuffled_kl_window0": control[name]}
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["candidates"], indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4], dict(a.split("=", 1) for a in sys.argv[5:]))
