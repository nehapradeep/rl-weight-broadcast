# debug_uccl_port.py
import os
os.environ["UCX_NET_DEVICES"] = "mlx5_1:1"
os.environ["UCX_IB_GID_INDEX"] = "3"
os.environ["UCCL_NET_DEVICES"] = "mlx5_1"

import torch
torch.cuda.set_device(0)

from uccl import p2p
import subprocess
import time

print("=" * 60)
print("BEFORE creating endpoint:")
result = subprocess.run(["ss", "-tuln"], capture_output=True, text=True)
print(result.stdout)

print("=" * 60)
print("Creating endpoint...")
ep = p2p.Endpoint(0, 16)

metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(metadata)

print(f"\nUCCL reports:")
print(f"  IP:   {ip}")
print(f"  PORT: {port}")
print(f"  GPU:  {gpu}")

time.sleep(2)

print("=" * 60)
print("AFTER creating endpoint:")
result = subprocess.run(["ss", "-tuln"], capture_output=True, text=True)
print(result.stdout)

print("=" * 60)
print(f"Looking for port {port}...")
if str(port) in result.stdout:
    print(f"  ✓ Found TCP listener on port {port}")
else:
    print(f"  ✗ NO TCP listener on port {port}")
    print(f"  → UCCL is using RDMA CM, not TCP")

# Check RDMA CM
print("\n" + "=" * 60)
print("Checking RDMA connections:")
result = subprocess.run(["rdma", "res", "show", "cm_id"], capture_output=True, text=True)
print(result.stdout if result.stdout else "No RDMA CM connections")

print("\nServer ready. Waiting for connection...")
ok, r_ip, r_gpu, conn_id = ep.accept()
print(f"Result: ok={ok}")