#!/usr/bin/env bash
# Full benchmark: vllm bench serve across concurrency/lengths + GPU telemetry.
# Results land in ~/twinspark/bench-results/
set -uo pipefail
OUT=~/twinspark/bench-results
mkdir -p $OUT

run_bench() { # $1=tag $2=in_len $3=out_len $4=concurrency $5=num_prompts
  echo "### $1 (in=$2 out=$3 conc=$4 n=$5)"
  # GPU 遥测后台采样
  ( while :; do
      s1=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader 2>/dev/null)
      s2=$(ssh -o BatchMode=yes "${PEER_USER:-$USER}@${PEER_QSFP_IP:-10.0.0.2}" 'nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader' 2>/dev/null)
      echo "$(date +%s),$s1,$s2"; sleep 5
    done > $OUT/$1.telemetry.csv 2>/dev/null ) &
  TPID=$!
  sudo docker exec ds0731-head vllm bench serve \
    --host localhost --port 8000 \
    --model /models/DeepSeek-V4-Flash-0731 --served-model-name ds-0731 \
    --dataset-name random --random-input-len $2 --random-output-len $3 \
    --num-prompts $5 --max-concurrency $4 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir /ds0731/bench-results --result-filename $1.json \
    2>&1 | tail -30 | tee $OUT/$1.summary.txt
  kill $TPID 2>/dev/null
}

case "${1:-all}" in
  quick) run_bench c1_1k1k 1024 1024 1 8 ;;
  all)
    run_bench c1_1k1k   1024 1024 1 8
    run_bench c2_1k1k   1024 1024 2 12
    run_bench c4_1k1k   1024 1024 4 16
    run_bench c6_1k1k   1024 1024 6 18
    run_bench c1_8k1k   8192 1024 1 6
    run_bench c4_8k1k   8192 1024 4 12
    run_bench c1_64k512 65536 512 1 3
    ;;
esac
echo "结果在 $OUT/"
