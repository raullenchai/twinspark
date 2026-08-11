#!/usr/bin/env bash
# ds0731-head container PID1: ray head -> wait for 2 GPUs -> vLLM
# Any container restart (boot / docker restart) re-runs the full sequence => self-healing
set -u
mkdir -p /ds0731/logs
ray stop --force >/dev/null 2>&1 || true
until ray start --head --node-ip-address="${HEAD_IP}" --port=6379 --num-gpus=1 >/dev/null 2>&1; do
  echo "[boot-head] ray head failed to start, retrying in 5s"; sleep 5
done
echo "[boot-head] ray head up, waiting for worker to join (need 2 GPUs)..."
while :; do
  G=$(python3 - <<'PY' 2>/dev/null | tail -1
import ray
try:
    ray.init(address="auto", log_to_driver=False)
    print(int(ray.cluster_resources().get("GPU", 0)))
    ray.shutdown()
except Exception:
    print(0)
PY
)
  [ "${G:-0}" = "2" ] && break
  sleep 5
done
echo "[boot-head] 2 GPUs ready, starting vLLM ($(date))"
exec bash /ds0731/launch/launch-vllm.sh >> /ds0731/logs/vllm.log 2>&1
