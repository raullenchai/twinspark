# twinspark — DeepSeek-V4-Flash-0731 (284B MoE) on 2× NVIDIA DGX Spark

*Production-grade recipe: self-healing 2-node vLLM cluster, reboot-verified, tuned DSpark speculative decoding (~75 tok/s), full benchmarks, and OpenAI Codex CLI integration. Snapshot as of 2026-08-11 — versions are pinned; expect drift.*

## TL;DR

- **Model**: `deepseek-ai/DeepSeek-V4-Flash-0731` — 284B MoE (13B active), native FP4 experts + FP8 attention, 1M context, MIT license. 167 GB of safetensors.
- **Hardware**: 2× DGX Spark (GB10, 121 GiB unified memory each, 4 TB NVMe), linked by a single 400G QSFP DAC cable → shows up as **2× 200GbE RoCE links**.
- **Engine**: vLLM 0.25.2 (community image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` with GB10/sm_121 kernels: FlashInfer-B12X MoE, NVFP4 DS-MLA KV cache, DSpark speculative decoding) + Ray for 2-node TP=2.
- **Results**: **~75 tok/s single-stream decode** (DSpark speculative decoding = 2.74× over the 27 tok/s bandwidth floor), ~1.7k tok/s prefill, 1M context enabled, KV capacity 2.74M tokens. Full benchmark tables below.
- **Daily driver**: OpenAI Codex CLI pointed at the local endpoint (`wire_api = "responses"`) — agentic coding evaluated on 3 greenfield projects **plus one real feature landed in a 1,831-file codebase** (results below).

---

## Quick start

Assumes both Sparks are through DGX OS first-boot, the QSFP DAC is plugged (before power-on!), and you can SSH between them.

```bash
# 0. on BOTH nodes: RDMA prerequisites (memlock etc.) — see §2 gotchas #1-2
# 1. on BOTH nodes: install the netplan file (swap IPs on node 2)
sudo cp netplan/99-qsfp.yaml /etc/netplan/ && sudo chmod 600 /etc/netplan/99-qsfp.yaml && sudo netplan apply

# 2. verify the fabric BEFORE touching any model (do not skip):
scripts/preflight.sh            # links at 200000Mb/s, MTU 9000, RoCE ACTIVE
#    then nccl-tests all_reduce_perf across nodes — expect ~20 GB/s bus BW (§1)

# 3. get the weights on node 1, rsync to node 2 over the QSFP link
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir ~/models/DeepSeek-V4-Flash-0731

# 4. build the serving image on BOTH nodes (adds Ray to the anemll image):
docker build -t ds0731-vllm:ray - <<'DF'
FROM ghcr.io/anemll/dspark-vllm-gx10:0.1.1
RUN pip install --no-cache-dir "ray[default]"
DF

# 5. edit scripts/twinspark.env (your IPs/usernames), then from node 1:
scripts/cluster-start.sh        # ~8 min to serving
scripts/status.sh
curl -s localhost:8000/v1/models | jq -r '.data[0].id'   # -> ds-0731
```

## 1. Hardware & Topology

```
            ┌────────── LAN switch (management, 2.5GbE) ──────────┐
            │                                                     │
         spark1 (192.168.1.10)                          spark2 (192.168.1.11)
            │  ConnectX-7  ═══ 400G QSFP DAC (0.5 m) ═══  ConnectX-7
            │   10.0.0.1 / 10.0.1.1        10.0.0.2 / 10.0.1.2
            └────── RoCE v2, MTU 9000, NCCL over 2 rails ─────┘
```

Each DGX Spark: GB10 (Grace 20-core ARM + Blackwell GPU, sm_121), 128 GB unified LPDDR5x (121 GiB usable), ~273 GB/s memory bandwidth, 4 TB NVMe, DGX OS 7.2.3 (Ubuntu 24.04 arm64), CUDA 13.0, driver 580.173.02.

### Discovery #1: one cable, two NICs

The single QSFP cage is internally wired to **two independent ConnectX-7 controllers**. Plugging one 400G DAC lights up **two** 200G links (`enp1s0f0np0` + `enP2p1s0f0np0`, RDMA `rocep1s0f0` + `roceP2p1s0f0`). Each controller sits on a **PCIe Gen5 x4** link (~126 Gb/s usable), so per-rail RDMA tops out at ~109 Gb/s (`ib_write_bw`, 87% of the PCIe ceiling — the NIC is not the bottleneck, the PCIe lane is).

Give **both** rails to NCCL:

```bash
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=enP7s7      # bootstrap over the management LAN
```

Measured cross-node `all_reduce_perf` (nccl-tests, NCCL 2.30.7): **21.7 GB/s bus bandwidth** peak at 1 GB messages, 17.8 GB/s average. If you see ~2 GB/s, NCCL silently fell back to TCP — fix your RDMA before loading any model.

### Discovery #2: the CX-7 is power-gated

With no cable plugged, DGX OS powers the CX-7 down and removes it from the PCI bus entirely (dmesg shows `Detected insufficient power on the PCIe slot (27W)` then the devices vanish). Don't panic when `lspci` shows nothing — **plug the cable before boot** (or `echo 1 > /sys/bus/pci/rescan`).

### Netplan (`/etc/netplan/99-qsfp.yaml`, mirrored with .1/.2 swapped)

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:    { addresses: [10.0.0.1/24], mtu: 9000, dhcp4: false, dhcp6: false, optional: true }
    enP2p1s0f0np0:  { addresses: [10.0.1.1/24], mtu: 9000, dhcp4: false, dhcp6: false, optional: true }
```

MTU 9000 verified end-to-end (`ping -M do -s 8972`), RoCE `active_mtu` goes 1024 → 4096.

---

## 2. Gotchas that cost us real time (read this before you start)

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | NCCL `unhandled system error`, `ncclIbRegMrDmaBufInternal -> 2` | default `memlock` ulimit is **8 MB**; RDMA can't register memory | `* soft/hard memlock unlimited` in `/etc/security/limits.d/99-rdma.conf` + `DefaultLimitMEMLOCK=infinity` for systemd + `--ulimit memlock=-1` on every container |
| 2 | limits.d fix "doesn't work" | **Tailscale SSH bypasses pam_limits** — sessions through `tailscale up --ssh` keep the old 8 MB | test via OpenSSH (`ssh localhost`), not Tailscale SSH |
| 3 | Ray worker dies during model load, `killed due to memory pressure (OOM)` while 117 GiB is free | DGX Spark has **unified memory** — vLLM's "GPU" allocation counts as host RAM; Ray's memory monitor kills the worker | `RAY_memory_monitor_refresh_ms=0` on **both** nodes |
| 4 | Model loads, `/v1/completions` is perfect, but **chat output is nondeterministic token salad with random Chinese** | DSpark speculative decoding: rejected draft tokens leave dirty KV context; plus vLLM auto-enables breakable CUDA graphs | `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` and `VLLM_USE_BREAKABLE_CUDAGRAPH=0` |
| 5 | `--speculative-config` JSON mangled (`Value {method:dspark} cannot be converted`) | quoting through nested `bash -lc "..."` layers | never inline the launch command — ship a **script file** into the container and exec it |
| 6 | vLLM never starts, log file absent | `pkill -f 'vllm serve'` matches the wrapper shell's own command line and kills itself | `pkill -f 'bin/vllm'` |
| 7 | Official `vllm/vllm-openai:cu130-nightly` can't load the model | it's pinned to vLLM 0.19.x — predates `DeepseekV4ForCausalLM` | community image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2 + sm_121 kernels) |
| 8 | vLLM stable arm64 images crash on GB10 | bundled torch compiled through sm_120 only; GB10 is **sm_121** | same as #7 |
| 9 | Codex CLI: `wire_api = "chat"` rejected | removed in codex 0.147 | `wire_api = "responses"` — vLLM 0.25 ships `/v1/responses` |
| 10 | Codex warns `Model metadata for ds-0731 not found` | unknown model → fallback metadata | undocumented `model_catalog_json` key + catalog JSON (schema below) |
| 11 | After restarting the head container, it hangs forever "waiting for 2 GPUs" | Ray head restart = **new GCS session**; the worker's old raylet never re-registers, yet `ray status` from the worker still succeeds against the new GCS → naive watchdogs never fire | worker watchdog must verify **its own IP is in `ray.nodes()` alive list**; param-change restarts bounce worker first, then head |
| 12 | Agent sessions degrade at ~100–140k context: the model narrates intentions in a loop ("Let me grep… Let me view…") or repeats identical tool calls, while the server is perfectly healthy | **Model-side, not your rig**: DeepSeek's own tech report shows retrieval stable only to **128K**; past it, self-reinforcing in-context repetition takes over. Reproduced on official DashScope serving ([qwen-code #4695](https://github.com/QwenLM/qwen-code/issues/4695) — 43 identical tool calls while the model outputs "Let me stop this loop") | break the loop with `/compact`; prevent it with Codex `model_auto_compact_token_limit = 120000` (sits just above the observed 100–120k onset; drop to 100k if loops persist). 1M context is real for *reading*, not for *accumulated agent history* |

---

## 3. Software stack

| Layer | Version |
|---|---|
| DGX OS | 7.2.3 (Ubuntu 24.04.4, kernel 6.17.0-1029-nvidia, aarch64) |
| Driver / CUDA | 580.173.02 / 13.0 |
| Container | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` + `pip install ray[default]` (ray 2.57.0), rebuilt as `ds0731-vllm:ray` |
| vLLM | 0.25.2.dev0+g752a3a504 (DeepseekV4ForCausalLM, DSpark, nvfp4_ds_mla KV, flashinfer_b12x MoE) |
| NCCL | 2.30.7+cuda13.3 |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` (48 shards, 167 GB, native FP4/FP8 — **this is the highest precision that exists**; don't waste time on GGUF re-quants) |
| Codex CLI | 0.147.0 (aarch64-musl static binary) |

Weights are transferred to node 2 over the QSFP link (`rsync` @ ~440 MB/s, ssh-crypto-bound, 380 s) — don't download twice.

---

## 4. Architecture: self-healing containers

Both machines run one container each with `--restart unless-stopped`. The container entrypoints implement the full recovery logic, so **a reboot of either or both machines auto-recovers to a serving state** with zero manual steps (~8 min to ready):

- **spark1 / `ds0731-head`**: `scripts/boot-head.sh` → start Ray head → wait until cluster has 2 GPUs → `exec launch-vllm.sh` (vLLM as PID 1; if it crashes, the container restarts and the whole sequence re-runs).
- **spark2 / `ds0731-worker`**: `scripts/boot-worker.sh` → join Ray head (retry loop) → watchdog: if head disappears, exit → docker restarts the container → rejoin fresh.
- All launch parameters live in **one file on the host** (`launch/launch-vllm.sh`, bind-mounted). Change params → `restart-vllm.sh` (= `docker restart ds0731-head`; worker follows automatically).

### Final launch configuration (`launch/launch-vllm.sh`)

```bash
vllm serve /models/DeepSeek-V4-Flash-0731 \
  --served-model-name ds-0731 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --kv-cache-dtype nvfp4_ds_mla \
  --moe-backend flashinfer_b12x \
  --block-size 256 \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 24 \
  --generation-config vllm \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

### Container environment (both nodes)

```
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0   NCCL_NET_GDR_LEVEL=5
NCCL_SOCKET_IFNAME=enP7s7             GLOO_SOCKET_IFNAME=enP7s7
RAY_memory_monitor_refresh_ms=0       RAY_DISABLE_MEMORY_MONITOR=1
VLLM_USE_FLASHINFER_SAMPLER=1         VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_WO_PROJECTION=1         VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0        DEFAULT_THINKING=low
HF_HUB_OFFLINE=1
docker: --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
        --cap-add=IPC_LOCK --device=/dev/infiniband --gpus all
```

Memory budget after load: model 2× 79.2 GiB, KV cache 20.5 + 18.4 GiB → **2,737,768 tokens of KV capacity** (nvfp4_ds_mla, 584-byte padded sparse-MLA envelope).

---

## 5. Speculative decoding tuning

DSpark = DeepSeek's built-in MTP draft module (the extra ~20B params in the 304B checkpoint). Acceptance rate measured ~75% over 5 draft positions (pos0 93%, declining to ~60% at pos4).

| config | single-stream tok/s (mean of 3) | 4-way concurrent agg | 7k-prompt TTFT | prefill tok/s |
|---|---|---|---|---|
| spec decode OFF | 27.3 (27.3/27.3/27.3 — the pure bandwidth floor) | 71.3 | 4.08 s | 1745 |
| num_speculative_tokens=3 | 64.0 | **100.2** | 4.12 s | 1728 |
| num_speculative_tokens=5 (recipe default) | 70.7 | 64.1 | 4.7 s | 1517 |
| **num_speculative_tokens=7 ← winner** | **74.8** (74.4–75.2) | 82.6 | 4.26 s | 1674 |

Takeaways:

- **DSpark = 2.74× single-stream speedup** (27.3 → 74.8 tok/s). The no-spec floor of 27.3 tok/s is exactly what the memory-bandwidth math predicts for 13B active @ ~4.4 bit over 2× 273 GB/s with per-layer all-reduce over the QSFP link.
- Acceptance stays high enough (~75% over 5 positions) that deeper drafts keep paying: 7 beats 5 beats 3 for a single stream.
- **Speculative decoding inverts at concurrency**: at 4 concurrent streams, spec3 wins (100 tok/s agg) and spec5/7 lose to plain decoding's batching. Tune for your workload — we serve 1–2 Codex streams, so 7 it is.

### Addendum: multi-agent production tuning (measured after a day of real use)

Running 2+ Codex agents against the endpoint revealed two things the clean-prompt numbers hide:

1. **Agentic traffic is prefill-dominated.** Every tool round re-submits a large context; we observed ~8k tok/s of prompt processing sustained while decode sat at ~51 tok/s aggregate. Prefix caching absorbs most of the cost, but prefill still competes with decode under chunked prefill.
2. **DSpark acceptance drops from ~75% to ~40% on agent contexts** — draft models predict clean prose/code well, but not tool output (file listings, test logs). Deep drafts waste compute exactly when you're busiest.

So we ship two launch profiles (`launch/`):

| profile | config | single-stream | 4-way agg | prefill | 7k TTFT |
|---|---|---|---|---|---|
| `launch-vllm.sh` (default) | spec7, seqs 6 | **74.8 tok/s** | 82.6 | 1674 | 4.26 s |
| `launch-vllm-multiagent.sh` | spec3, seqs 8 | 66.3 | **105.7** | **1953** | **3.65 s** |

Swap = copy over `launch-vllm.sh` + run `scripts/restart-vllm.sh` (~8 min).

**Winner: `num_speculative_tokens=7`** → baked into `launch/launch-vllm.sh`.

---

## 6. Full benchmark (`vllm bench serve`, random dataset)

| scenario | conc | output tok/s | total tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
|---|---|---|---|---|---|
| 1k in / 1k out | 1 | 33.8 | 67.5 | 683 / 706 ms | 29.6 / 38.4 ms |
| 1k / 1k | 2 | 51.6 | 103.3 | 466 / 829 ms | 31.4 / 55.6 ms |
| 1k / 1k | 4 | 56.9 | 113.8 | 610 / 937 ms | 69.4 / 100.2 ms |
| 1k / 1k | 6 | 72.0 | 144.0 | 716 / 1200 ms | 74.6 / 109.9 ms |
| 8k / 1k | 1 | 20.0 | 180.2 | 5.1 / 5.5 s | 50.3 / 56.4 ms |
| 8k / 1k | 4 | 39.5 | 355.7 | 2.3 / 5.9 s | 98.8 / 121.1 ms |
| **64k / 512** | 1 | 6.4 | 827.8 | **42 / 84 s** | 50.2 / 51.2 ms |

**Important caveat — random-token inputs sandbag speculative decoding.** `vllm bench serve --dataset-name random` feeds random token soup, so the DSpark draft model can't predict continuations and acceptance collapses: 33.8 tok/s single-stream here vs **74.8 tok/s on real coding prompts** (our `measure.py`, temp 0.2). Real agentic workloads run at the higher number; the table above is the pessimistic floor. TPOT p50 of 29.6 ms @ conc 1 also implies ~34 tok/s worst-case per stream.

Long context: 64k-token prompts prefill at ~1.55k tok/s (42 s TTFT) and decode speed holds at 50 ms/token — the sparse-MLA indexer doing its job. 1M context is configured and KV capacity (2.74 M tokens) covers it, but budget ~11 min of prefill for a full 1M prompt.

**Thermals under sustained load**: worst case across all runs — spark1 79 °C / 73 W, spark2 78 °C / 69 W (64k-prefill group). Fans audible but far from throttling. Idle: ~43 °C / 12 W each.

---

## 7. Codex CLI as the daily-driver client

### Config that works (codex 0.147)

`~/.codex/config.toml`:

```toml
model = "ds-0731"
model_provider = "spark"
model_catalog_json = "/home/USER/.codex/ds0731-model-catalog.json"

[model_providers.spark]
name = "DeepSeek V4 Flash 0731 (local vLLM)"
base_url = "http://localhost:8000/v1"   # or the Tailscale IP from other machines
wire_api = "responses"                  # "chat" was removed in codex 0.147
env_key = "VLLM_API_KEY"                # export VLLM_API_KEY=dummy (vLLM has no auth)
```

The catalog JSON (undocumented but shipped feature) registers context window (1,048,576), tool support, and base instructions for the unknown model — kills the "fallback metadata" warning. Also set `model_auto_compact_token_limit = 120000` — see gotcha #12: agent sessions loop past ~128k context, and auto-compaction is the fence. Ubuntu 24.04 also needs `kernel.apparmor_restrict_unprivileged_userns=0` for codex's bubblewrap sandbox.

### Agentic evaluation: 3 medium projects, unattended

We gave `codex exec` (workspace-write sandbox, no human in the loop) three medium-difficulty specs and independently verified everything afterward — running the test suites ourselves, hitting the live services with real traffic, and writing adversarial tests the model never saw.

| project | spec highlights | wall time | tokens | independent verification | supervision rounds |
|---|---|---|---|---|---|
| **Expense tracker CLI** (Python/sqlite3/argparse) | 5 subcommands, budgets with exit-code semantics, CSV export, pytest | 111 s (+51 s fix round) | 35k | 5/5 tests green; CLI acceptance & CSV verified by hand | **1** — printed the over-budget warning but exited 0 where the spec required exit code 2; fixed correctly from a plain-English bug report in 51 s |
| **URL shortener API** (FastAPI/sqlite3) | base62 codes, custom aliases + 409, 307 redirects, click stats, admin-token delete, per-IP sliding-window rate limit, pytest | 564 s | 62k | 6/6 tests green; live uvicorn session: shorten/redirect/stats/409/422/404/401 + rate-limit 429 all correct | **0** |
| **Rate limiter library** (pure stdlib, thread-safe) | TokenBucket + SlidingWindowLog + decorator, concurrency no-over-grant test | 104 s | 15k | 10/10 tests ×3 runs; 6 adversarial checks (burst cap, 8-thread over-grant, blocking-timeout semantics, key isolation, expiry, decorator raise) all pass | **0** |

**Verdict**: 2 of 3 projects were correct zero-shot end-to-end; the third needed exactly one bug-report iteration. ~13 minutes of total agent wall time for three working, tested projects. Subjectively: it reads specs carefully, iterates until tests pass, writes clean idiomatic Python, and its self-reports were accurate (everything it claimed passed did pass — it just missed one exit-code requirement). At 75 tok/s the interactive feel is comparable to a fast cloud API. This is a genuinely usable local daily-driver for agentic coding.

### The hard test: a real feature in a 1,831-file codebase

Small greenfield projects don't separate frontier models from strong mid-tier ones — big codebases do. So we pointed the same unattended `codex exec` at a fresh clone of [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) (1,831 files, a **17,392-test** suite, green baseline) and asked for a real roadmap feature (issues #1335 + #319): **multi-model serving behind one endpoint with lazy load, single-flight, idle unload, and LRU eviction** — an architecture-level concurrency feature touching CLI, config, server assembly, and four route families.

**Round 1 (unattended, 27 min, 278k tokens, 152 tool calls):** it shipped a genuinely well-designed implementation — a self-contained 492-line pool module with an injectable clock for testability, per-model `asyncio.Future` single-flight, and even the subtle `asyncio.wait_for(asyncio.shield(future))` idiom so one waiter's timeout can't cancel a shared load (a detail human reviewers regularly get wrong). 27 new tests, all green; `DESIGN.md` included; honest disclosure of what it couldn't verify. **But**: it renamed a module-level seam (`routes.chat.get_engine`) that 200 existing tests monkeypatch — tests it *couldn't run* because its sandbox has no Metal — and shipped the regression: **200 failed / 16,707 passed**.

**Round 2 (we played CI, 16 min, 207k tokens):** given only the failing-test list and one sample traceback — no root-cause hint, still no ability to run the affected tests itself — it diagnosed the seam breakage from source, restored the interface while keeping the feature intact, and left the codebase at **16,900+ tests fully green** with its 27 new tests still passing.

**Long-horizon verdict:** one supervision round to land a real concurrency feature in a large codebase. The failure mode was interesting and very human: not hallucination, not spaghetti — it broke a contract it had no way to see, and disclosed the blind spot in its own report. Our honest placement after both evals: **frontier-competitive on well-specified, self-contained work; roughly one supervision round behind top frontier models on large-codebase integration work** — which, for 13 B active parameters running on two desk-side boxes, is remarkable.

---

## 8. Ops runbook

```bash
scripts/status.sh        # one-glance health: containers, vLLM, GPUs, QSFP
scripts/restart-vllm.sh  # apply new params from launch-vllm.sh (~8 min)
scripts/cluster-start.sh # (re)create both containers from scratch
scripts/cluster-stop.sh  # stop everything
tail -f ~/twinspark/logs/vllm.log
curl -s localhost:8000/metrics | grep spec_decode   # DSpark acceptance telemetry
```

Reboot behavior: power-cycle either or both machines → docker restarts containers → Ray re-forms → vLLM re-serves. **Verified by actually rebooting both nodes simultaneously**: SSH back in 40 s, Ray cluster re-formed ~2.5 min after boot, vLLM serving again ~9 min after power-on — zero manual steps.

---

## Repo layout

```
scripts/   cluster lifecycle: twinspark.env (edit me) · cluster-start/stop · boot-head/worker · restart-vllm · status · preflight
launch/    launch-vllm.sh — the single source of truth for all vLLM parameters
bench/     measure.py (real-prompt perf) · bench.sh (vllm bench serve suite) · results/*.json (raw data)
codex/     Codex CLI config + model-catalog examples
netplan/   QSFP link template
```

*Setup, debugging, tuning and this write-up were done by Claude (Anthropic) driving the cluster end-to-end. Hardware: 2× DGX Spark. Time from first boot to serving: about one evening, including hitting every gotcha above. Pinned to the versions in §3 — PRs updating for newer stacks welcome, but this repo is a snapshot, not a maintained product.*
