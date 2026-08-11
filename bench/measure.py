#!/usr/bin/env python3
"""Unified perf measurement: single-stream decode x3 / 4-way concurrent agg / 8k-prompt TTFT."""
import httpx, json, time, sys, concurrent.futures as cf

BASE = "http://localhost:8000/v1"
CODE_PROMPT = "Write a complete Python implementation of an LRU cache with TTL support, generics-style type hints, and unit tests. Output code only."
PROSE_PROMPT = "Explain how a modern SSD works: NAND flash, FTL, wear leveling, garbage collection, SLC cache. Be thorough."

def chat(prompt, max_tokens, temperature=0.2, timeout=600):
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}/chat/completions", json={
        "model": "ds-0731",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature}, timeout=timeout)
    el = time.monotonic() - t0
    d = r.json()
    u = d["usage"]
    return u["completion_tokens"], el, u["prompt_tokens"], d["choices"][0]["message"]["content"]

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    out = {"label": label}

    # warmup
    chat("Say OK.", 8)

    # 1) single-stream decode x3
    singles = []
    for i in range(3):
        ct, el, _, _ = chat(CODE_PROMPT, 600)
        singles.append(ct/el)
    out["single_stream_tok_s"] = [round(x,1) for x in singles]
    out["single_mean"] = round(sum(singles)/len(singles),1)

    # 2) 4-way concurrent aggregate
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(chat, p, 400) for p in [CODE_PROMPT, PROSE_PROMPT]*2]
        results = [f.result() for f in futs]
    el = time.monotonic() - t0
    total = sum(r[0] for r in results)
    out["concurrent4_agg_tok_s"] = round(total/el,1)
    out["concurrent4_per_stream"] = round(total/el/4,1)

    # 3) 8k-prompt TTFT (approximated with max_tokens=1)
    long_prompt = ("The quick brown fox jumps over the lazy dog. " * 800)[:32000] + "\nSummarize the above in 5 words."
    ct, el, pt, _ = chat(long_prompt, 1, timeout=900)
    out["prefill_prompt_tokens"] = pt
    out["prefill_ttft_s"] = round(el,2)
    out["prefill_tok_s"] = round(pt/el,0)

    # 4) correctness sentinel (temp 0, must contain def)
    _,_,_, content = chat("Write a Python function that returns the n-th prime using a sieve. Code only.", 150, 0)
    out["sanity_code_ok"] = ("def" in content and "prime" in content.lower())

    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
