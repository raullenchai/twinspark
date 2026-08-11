#!/usr/bin/env bash
# (Re)create the two self-healing containers. Launch params live in launch/launch-vllm.sh.
set -euo pipefail
cd "$(dirname "$0")"; source ./twinspark.env

RAY_ENV="-e RAY_memory_monitor_refresh_ms=0 -e RAY_DISABLE_MEMORY_MONITOR=1"
NCCL_ENV="-e NCCL_IB_HCA=$NCCL_HCAS -e NCCL_NET_GDR_LEVEL=5 -e NCCL_SOCKET_IFNAME=$MGMT_IFACE -e NCCL_IB_DISABLE=0 -e NCCL_DEBUG=WARN -e GLOO_SOCKET_IFNAME=$MGMT_IFACE"
VLLM_ENV="-e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e DEFAULT_THINKING=low -e HF_HUB_OFFLINE=1"
LIMITS="--ulimit memlock=-1 --ulimit stack=67108864 --cap-add=IPC_LOCK --device=/dev/infiniband"

echo "### sync scripts to spark2"
rsync -a --exclude logs "$(pwd)/../" "$PEER_USER@$PEER_QSFP_IP:~/twinspark/"

echo "### spark1: head (self-healing: ray head -> wait 2 GPUs -> vLLM)"
sudo docker rm -f ds0731-head >/dev/null 2>&1 || true
sudo docker run -d --name ds0731-head --restart unless-stopped \
  --gpus all --network host --ipc=host $LIMITS $NCCL_ENV $RAY_ENV $VLLM_ENV \
  -e VLLM_HOST_IP="$HEAD_LAN_IP" -e HEAD_IP="$HEAD_LAN_IP" \
  -v "$MODEL_DIR:/models/DeepSeek-V4-Flash-0731:ro" \
  -v "$HOME/twinspark:/ds0731" \
  --entrypoint bash "$IMG" /ds0731/scripts/boot-head.sh

echo "### spark2: worker (self-healing: join loop + membership watchdog)"
ssh -o BatchMode=yes "$PEER_USER@$PEER_QSFP_IP" "sudo docker rm -f ds0731-worker >/dev/null 2>&1 || true; \
  sudo docker run -d --name ds0731-worker --restart unless-stopped \
  --gpus all --network host --ipc=host $LIMITS $NCCL_ENV $RAY_ENV $VLLM_ENV \
  -e VLLM_HOST_IP=$PEER_LAN_IP -e HEAD_ADDR=$HEAD_LAN_IP:6379 \
  -v \$HOME/models/DeepSeek-V4-Flash-0731:/models/DeepSeek-V4-Flash-0731:ro \
  -v \$HOME/twinspark:/ds0731 \
  --entrypoint bash $IMG /ds0731/scripts/boot-worker.sh"

echo "### done. vLLM log: tail -f ~/twinspark/logs/vllm.log  (ready in ~8 min)"
