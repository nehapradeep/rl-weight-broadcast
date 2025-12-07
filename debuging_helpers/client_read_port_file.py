# client_read_port_file.py
import os
os.environ["UCX_NET_DEVICES"] = "mlx5_1:1"
os.environ["UCX_IB_GID_INDEX"] = "3"
os.environ["UCCL_NET_DEVICES"] = "mlx5_1"

import torch
torch.cuda.set_device(0)
from uccl import p2p
import time

# Wait for server to write port file
# Copy file from Node 3 or use shared filesystem
port_file = "/tmp/uccl_server_port.txt"

print(f"Reading server info from {port_file}...")
# You need to copy this file from Node 3, or manually enter values
# scp node3:/tmp/uccl_server_port.txt /tmp/

with open(port_file, "r") as f:
    ip, port, gpu = f.read().strip().split(",")
    port = int(port)
    gpu = int(gpu)

print(f"Server: IP={ip}, Port={port}, GPU={gpu}")

print("Creating endpoint...")
ep = p2p.Endpoint(0, 16)

print(f"Connecting to {ip}:{gpu} via port {port}...")
ok, conn_id = ep.connect(ip, gpu, remote_port=port)

if ok:
    print(f"✓ SUCCESS! conn_id={conn_id}")
else:
    print("✗ FAILED")