#!/usr/bin/env bash
# ds0731-worker container PID1: join head in a loop; exit when this node leaves the
# cluster -> docker restarts container -> rejoin fresh.
# The watchdog MUST check "is my IP in the alive-nodes list", NOT `ray status`:
# after a head restart (new GCS session) `ray status` still succeeds even though
# this node is no longer a member.
set -u
MY_IP="${VLLM_HOST_IP}"
ray stop --force >/dev/null 2>&1 || true
until ray start --address="${HEAD_ADDR}" --num-gpus=1 >/dev/null 2>&1; do
  echo "[boot-worker] join ${HEAD_ADDR} failed, retrying in 5s"; sleep 5
  ray stop --force >/dev/null 2>&1 || true
done
echo "[boot-worker] joined ray cluster ($(date))"
while :; do
  sleep 30
  IN=$(python3 - <<PY 2>/dev/null | tail -1
import ray
try:
    ray.init(address="${HEAD_ADDR}", log_to_driver=False)
    ips = [n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]]
    print(1 if "${MY_IP}" in ips else 0)
    ray.shutdown()
except Exception:
    print(0)
PY
)
  if [ "${IN:-0}" != "1" ]; then
    echo "[boot-worker] this node left the cluster (head restarted?), exiting to trigger rejoin"
    exit 1
  fi
done
