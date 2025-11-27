# rollout/rollout.py
import os, socket, torch, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.wire import recv_tensor

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "50051"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[rollout] connecting to {HOST}:{PORT} on {DEVICE} ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("[rollout] connected")

try:
    while True:
        t = recv_tensor(s, device=DEVICE)
        print(f"[rollout] got tensor: shape={tuple(t.shape)}, sum={t.sum().item():.3f}, device={t.device}")
except (ConnectionError, KeyboardInterrupt):
    print("[rollout] done")
finally:
    s.close()

