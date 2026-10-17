#!/usr/bin/env python3
"""Build dist/hardware.json from data/registry/hardware/: every GPU and chip we have a spec sheet for (with or
without a recipe), as the rows its page shows: memory, compute, throughput per precision, price, and the sources
each came from. `--check` fails if it is stale. Standard library only."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "registry" / "hardware"
OUT = ROOT / "dist" / "hardware.json"

COMPUTE = {  # scalar spec fields, in the order a page lists them
    "architecture": "Architecture", "graphics_model": "Graphics", "cuda_cores": "CUDA cores", "tensor_cores": "Tensor cores",
    "rt_cores": "RT cores", "compute_units": "Compute units", "graphics_compute_units": "Graphics compute units",
    "stream_processors": "Stream processors", "matrix_cores": "Matrix cores", "ai_accelerators": "AI accelerators",
    "xe_cores": "Xe cores", "xmx_engines": "XMX engines", "cpu_cores": "CPU cores", "tensor_ai_tops": "AI TOPS",
    "overall_ai_tops": "AI TOPS (whole chip)", "npu_tops": "NPU TOPS", "fp4_tops": "FP4 TOPS", "int8_tops": "INT8 TOPS", "int4_tops": "INT4 TOPS",
}
PRECISIONS = ["fp4", "fp8", "int8", "fp16", "bf16", "tf32", "fp32"]
PRICES = {"msrp": "MSRP", "current_street_price": "Street price", "current_system_price": "System price", "exact_configuration_price": "Configuration price"}
AVAIL = {"available": "In stock", "limited": "Limited", "unavailable": "Not available", "discontinued": "Discontinued"}


def num(v):
    return f"{v:,.0f}" if isinstance(v, (int, float)) and v >= 100 else f"{v:g}" if isinstance(v, float) else str(v)


def money(p):
    if not isinstance(p, dict) or p.get("amount") is None:
        return None
    a = f"{p['amount']:,.0f}" if float(p["amount"]).is_integer() else f"{p['amount']:,.2f}"
    return f"${a}" if p.get("currency", "USD") == "USD" else f"{a} {p['currency']}"


def record(h):
    m, c, com, acc = h.get("memory") or {}, h.get("compute") or {}, h.get("commercial") or {}, h.get("accelerator") or {}
    memory = [["Memory", f"{m['vram_gb']} GB {m.get('vram_type') or ''}".strip()]]
    if m.get("bandwidth_gb_per_s"):
        memory.append(["Bandwidth", f"{num(m['bandwidth_gb_per_s'])} GB/s"])
    if m.get("cpu_memory_gb") and h.get("kind") == "discrete":
        memory.append(["Host memory", f"{m['cpu_memory_gb']} GB"])
    compute = []
    for k, label in COMPUTE.items():
        v = c.get(k)
        if v not in (None, ""):
            compute.append([label, num(v)])
    if isinstance(acc.get("gpu_cores"), dict):
        g = acc["gpu_cores"]
        compute.append(["GPU cores", f"{g['min']}–{g['max']}" if g.get("min") != g.get("max") else str(g["max"])])
    elif acc.get("gpu_cores"):
        compute.append(["GPU cores", str(acc["gpu_cores"])])
    if acc.get("neural_engine_cores"):
        compute.append(["Neural Engine cores", str(acc["neural_engine_cores"])])
    if acc.get("scope"):
        compute.append(["Form", acc["scope"]])
    tflops = []  # [precision, dense, with 2:4 sparsity]
    stats = c.get("stats") or {}
    for p in PRECISIONS:
        s = stats.get(p) or {}
        d, sp = s.get("dense") or {}, s.get("structured_2_4") or {}
        if d.get("state") == "known" or sp.get("state") == "known":
            tflops.append([p.upper(), d.get("value") if d.get("state") == "known" else None, sp.get("value") if sp.get("state") == "known" else None])
    price = [[label, money(com.get(k))] for k, label in PRICES.items() if money(com.get(k))]
    av = (com.get("availability") or {}).get("state")
    if av and av != "unknown":
        price.append(["Availability", AVAIL.get(av, av)])
    listings = sorted((p for p in com.get("prices") or [] if p.get("amount") is not None and (p.get("source") or {}).get("url")),
                      key=lambda p: str(p.get("as_of") or p.get("captured_at")), reverse=True)
    seen, sources = set(), []
    for s in (h.get("sources") or []) + [x for f in (h.get("facts") or {}).values() for x in (f.get("provenance") or {}).get("sources") or []]:
        if s.get("url") and s["url"] not in seen:
            seen.add(s["url"])
            sources.append({"publisher": s.get("publisher") or s.get("kind"), "url": s["url"], "at": (s.get("captured_at") or "")[:10]})
    return {
        "name": h["name"], "vendor": h["vendor"], "kind": h.get("kind"), "family": h.get("family"),
        "vram_gb": m.get("vram_gb"), "bandwidth_gb_s": m.get("bandwidth_gb_per_s"),
        "products": h.get("product_names") or [], "memory": memory, "compute": compute, "tflops": tflops, "price": price,
        "listings": [{"what": p.get("configuration") or p.get("kind", "").replace("_", " "), "price": money(p), "at": str(p.get("as_of") or "")[:10],
                      "publisher": p["source"].get("publisher"), "url": p["source"]["url"]} for p in listings[:6]],
        "sources": sources, "captured_at": h.get("captured_at"),
    }


def build():
    return {"schema": 1, "hardware": {f.stem: record(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(SRC.glob("*.json"))}}


if __name__ == "__main__":
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit("dist/hardware.json is stale: run python3 lab/hardware.py")
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"{OUT.relative_to(ROOT)}: {len(json.loads(text)['hardware'])} records")
