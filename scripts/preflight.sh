#!/usr/bin/env bash
# 双机链路预检 —— 跑模型之前必须全绿
set -u
echo "### 1) mlx5 网卡与速率"
for n in $(ls /sys/class/net); do
  d=$(basename "$(readlink -f /sys/class/net/$n/device/driver 2>/dev/null)" 2>/dev/null)
  [ "$d" = "mlx5_core" ] && { echo -n "  $n: "; sudo ethtool $n 2>/dev/null | grep -i "^\s*Speed" ; ip -4 -o addr show $n | awk '{print "     ip:",$4}'; ip -o link show $n | grep -o "mtu [0-9]*"; }
done
echo "### 2) RDMA 设备"
ibv_devinfo 2>/dev/null | grep -E "hca_id|state|link_layer|active_mtu" || echo "  无 RDMA 设备"
echo "### 3) 对端连通 (10.0.0.x)"
ping -c2 -W2 ${PEER_IB:-10.0.0.2} >/dev/null 2>&1 && echo "  ping OK" || echo "  ping FAIL"
echo "### 4) RDMA 带宽 (需对端先跑 ib_write_bw 服务端)"
echo "  对端: ib_write_bw -d \$HCA -F"
echo "  本端: ib_write_bw -d \$HCA -F ${PEER_IB:-10.0.0.2}"
echo "### 5) NCCL 跨机带宽 —— 最关键，必须 ~20+ GB/s"
echo "  见 ~/ds0731/nccl-check.sh"
