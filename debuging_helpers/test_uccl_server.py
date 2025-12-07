# test_server_debug.py
import os
os.environ["UCCL_DEBUG"] = "1"
os.environ["UCX_LOG_LEVEL"] = "debug"

import torch
torch.cuda.set_device(0)

from uccl import p2p

print("=" * 50)
print("UCCL P2P SERVER DEBUG")
print("=" * 50)

print("\n[1] Creating endpoint...")
ep = p2p.Endpoint(0, 16)

print("\n[2] Getting metadata...")
metadata = ep.get_metadata()
print(f"    Raw metadata length: {len(metadata)} bytes")
print(f"    Raw metadata (hex): {metadata.hex()[:100]}...")

print("\n[3] Parsing metadata...")
ip, port, gpu = p2p.Endpoint.parse_metadata(metadata)
print(f"    IP:   {ip}")
print(f"    Port: {port}")
print(f"    GPU:  {gpu}")

print("\n[4] Checking network interfaces...")
import subprocess
result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'inet ' in line or 'enp' in line or 'mlx' in line:
        print(f"    {line.strip()}")

print(f"\n[5] Server ready! Waiting for connection on {ip}:{port}...")
print("    (Run client on Node 0 now)")

ok, r_ip, r_gpu, conn_id = ep.accept()

if ok:
    print(f"\n[SUCCESS] Connection accepted!")
    print(f"    Remote IP:  {r_ip}")
    print(f"    Remote GPU: {r_gpu}")
    print(f"    Conn ID:    {conn_id}")
else:
    print("\n[FAILED] Connection not accepted")

print("=" * 50)