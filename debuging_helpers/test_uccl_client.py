# test_client_debug.py
import os
os.environ["UCCL_DEBUG"] = "1"
os.environ["UCX_LOG_LEVEL"] = "debug"

import torch
torch.cuda.set_device(0)

from uccl import p2p

# === UPDATE THESE VALUES FROM SERVER OUTPUT ===
SERVER_IP = "10.162.224.134"  # From server output
SERVER_PORT = 42279           # From server output  
SERVER_GPU = 0
# ==============================================

print("=" * 50)
print("UCCL P2P CLIENT DEBUG")
print("=" * 50)

print("\n[1] Creating endpoint...")
ep = p2p.Endpoint(0, 16)

print("\n[2] Getting local metadata...")
metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(metadata)
print(f"    Local IP:   {ip}")
print(f"    Local Port: {port}")
print(f"    Local GPU:  {gpu}")

print(f"\n[3] Target server:")
print(f"    Server IP:   {SERVER_IP}")
print(f"    Server Port: {SERVER_PORT}")
print(f"    Server GPU:  {SERVER_GPU}")

print("\n[4] Testing basic connectivity...")
import subprocess

# Ping test
result = subprocess.run(["ping", "-c", "1", "-W", "2", SERVER_IP], capture_output=True, text=True)
if result.returncode == 0:
    print(f"    Ping to {SERVER_IP}: SUCCESS")
else:
    print(f"    Ping to {SERVER_IP}: FAILED")

# Port test
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
try:
    sock.connect((SERVER_IP, SERVER_PORT))
    print(f"    TCP connect to {SERVER_IP}:{SERVER_PORT}: SUCCESS")
    sock.close()
except Exception as e:
    print(f"    TCP connect to {SERVER_IP}:{SERVER_PORT}: FAILED - {e}")

print(f"\n[5] Attempting UCCL P2P connection...")
print(f"    Connecting to {SERVER_IP}:{SERVER_GPU} via port {SERVER_PORT}...")

ok, conn_id = ep.connect(SERVER_IP, SERVER_GPU, remote_port=SERVER_PORT)

if ok:
    print(f"\n[SUCCESS] Connected!")
    print(f"    Conn ID: {conn_id}")
else:
    print("\n[FAILED] Connection failed")

print("=" * 50)