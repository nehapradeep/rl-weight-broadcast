# test_client_exact_port.py
import os
os.environ["UCX_NET_DEVICES"] = "mlx5_1:1"
os.environ["UCX_IB_GID_INDEX"] = "3"
os.environ["UCCL_NET_DEVICES"] = "mlx5_1"

import torch
torch.cuda.set_device(0)

from uccl import p2p

# EXACT values from server output
SERVER_IP = "10.162.224.134"
SERVER_PORT = 41289  # <-- EXACT port from server!
SERVER_GPU = 0

print(f"Connecting to {SERVER_IP}:{SERVER_GPU} via port {SERVER_PORT}...")

ep = p2p.Endpoint(0, 16)
ok, conn_id = ep.connect(SERVER_IP, SERVER_GPU, remote_port=SERVER_PORT)

if ok:
    print(f"✓ SUCCESS! conn_id={conn_id}")
else:
    print("✗ FAILED to connect")