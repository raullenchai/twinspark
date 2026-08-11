#!/usr/bin/env bash
# One-glance cluster health.
cd "$(dirname "$0")"; source ./twinspark.env
echo "== containers =="
sudo docker ps -a --filter name=ds0731 --format "  spark1/{{.Names}}: {{.Status}}"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER_USER@$PEER_QSFP_IP" 'sudo docker ps -a --filter name=ds0731 --format "  spark2/{{.Names}}: {{.Status}}"' 2>/dev/null || echo "  spark2: unreachable"
echo "== vLLM =="
if curl -s -m 3 localhost:8000/v1/models 2>/dev/null | grep -q ds-0731; then
  echo "  UP  http://$HEAD_LAN_IP:8000/v1  (model: ds-0731, 1M ctx)"
else
  echo "  DOWN. recent log:"
  tail -3 "$HOME/twinspark/logs/vllm.log" 2>/dev/null | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-120 | sed "s/^/  /"
fi
echo "== GPU =="
echo "  spark1: $(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader 2>/dev/null)"
echo "  spark2: $(ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER_USER@$PEER_QSFP_IP" 'nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader' 2>/dev/null)"
echo "== QSFP =="
for n in $(ls /sys/class/net); do
  d=$(basename "$(readlink -f /sys/class/net/$n/device/driver 2>/dev/null)" 2>/dev/null)
  [ "$d" = "mlx5_core" ] && echo "  $n: $(cat /sys/class/net/$n/operstate 2>/dev/null) $(sudo ethtool $n 2>/dev/null | grep -oE 'Speed: .*')"
done
