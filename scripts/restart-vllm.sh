#!/usr/bin/env bash
# Apply new params from launch/launch-vllm.sh: bounce worker first, then head
# (avoids the Ray GCS session-mismatch trap — see README gotcha #11).
set -euo pipefail
cd "$(dirname "$0")"; source ./twinspark.env
sudo truncate -s 0 "$HOME/twinspark/logs/vllm.log" 2>/dev/null || true
rsync -a --exclude logs "$(pwd)/../" "$PEER_USER@$PEER_QSFP_IP:~/twinspark/"
ssh -o BatchMode=yes "$PEER_USER@$PEER_QSFP_IP" 'sudo docker restart ds0731-worker' >/dev/null
sudo docker restart ds0731-head >/dev/null
echo "restarted, waiting for readiness:"
until curl -s -m 3 localhost:8000/v1/models 2>/dev/null | grep -q ds-0731; do sleep 10; printf .; done
echo " ready ($(date))"
