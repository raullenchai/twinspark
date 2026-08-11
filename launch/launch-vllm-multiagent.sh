#!/usr/bin/env bash
# Multi-agent variant: spec3 + max-num-seqs 8 — best when serving 3+ concurrent agents.
exec vllm serve /models/DeepSeek-V4-Flash-0731 \
  --served-model-name ds-0731 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
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
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 24 \
  --generation-config vllm \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
