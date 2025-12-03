# rollout/rollout.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import socket, torch, time
from utils.wire import recv_tensor

CTRL_HOST = os.getenv("CTRL_HOST", "127.0.0.1")
CTRL_PORT = int(os.getenv("CTRL_PORT", "50051"))
GPU  = int(os.getenv("GPU", "1"))
DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

print(f"[rollout] connecting to controller {CTRL_HOST}:{CTRL_PORT} on {DEVICE} ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((CTRL_HOST, CTRL_PORT))
s.sendall(b"ROLE rollout\n")
print("[rollout] connected")

recv_count = 0
t0 = time.time()
try:
    from transformers import GPT2Model

    model = GPT2Model.from_pretrained("gpt2").to(DEVICE)
    params = [p for p in model.parameters()]

    while True:
        flat_tensor = recv_tensor(s, device=DEVICE)
        recv_count += 1

        # copy the received flat weights into model parameters
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(flat_tensor[offset:offset+numel].view_as(p))
            offset += numel

        if recv_count % 10 == 0:
            dt = time.time() - t0
            print(f"[rollout] #{recv_count:04d} received GPT-2 weights (sum={flat_tensor.sum().item():.2f}, rate={recv_count/dt:.1f}/s)")

except (ConnectionError, KeyboardInterrupt):
    print("[rollout] done")
finally:
    s.close()

