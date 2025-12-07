# server_with_port_file.py
import os
os.environ["UCX_NET_DEVICES"] = "mlx5_1:1"
os.environ["UCX_IB_GID_INDEX"] = "3"
os.environ["UCCL_NET_DEVICES"] = "mlx5_1"

import torch
torch.cuda.set_device(0)
from uccl import p2p

print("Creating endpoint...")
ep = p2p.Endpoint(0, 16)

metadata = ep.get_metadata()
ip, port, gpu = p2p.Endpoint.parse_metadata(metadata)
print(f"Server: IP={ip}, Port={port}, GPU={gpu}")

# Save port to shared location (NFS/shared filesystem)
with open("/tmp/uccl_server_port.txt", "w") as f:
    f.write(f"{ip},{port},{gpu}")
print(f"Port saved to /tmp/uccl_server_port.txt")

print("Waiting for connection...")
ok, r_ip, r_gpu, conn_id = ep.accept()
print(f"Result: ok={ok}, conn_id={conn_id}")