# trainer/trainer.py
import os, time, socket, torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.wire import send_tensor

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "50051"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[trainer] listening on {HOST}:{PORT}, source device={DEVICE}")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

conn, addr = server.accept()
print(f"[trainer] rollout connected from {addr}")

# pretend “weights” that change over time
weights = torch.ones((1024, 1024), device=DEVICE)

try:
    step = 0
    while True:
        weights.add_(0.01)               # mutate in place
        send_tensor(conn, weights)       # push update
        step += 1
        if step % 10 == 0:
            print(f"[trainer] step {step}, sum={weights.sum().item():.2f}")
        time.sleep(0.5)                  # throttle a bit
except (BrokenPipeError, KeyboardInterrupt):
    print("[trainer] done")
finally:
    conn.close()
    server.close()

