#!/usr/bin/env bash
cd "$(dirname "$0")"; source ./twinspark.env
sudo docker rm -f ds0731-head 2>/dev/null || true
ssh -o BatchMode=yes "$PEER_USER@$PEER_QSFP_IP" "sudo docker rm -f ds0731-worker 2>/dev/null || true"
echo "both containers stopped"
