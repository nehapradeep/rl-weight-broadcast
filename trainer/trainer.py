import os, time, socket, torch, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.wire import send_tensor

CTRL_HOST = os.getenv("CTRL_HOST", "127.0.0.1")
CTRL_PORT = int(os.getenv("CTRL_PORT", "50051"))
GPU  = int(os.getenv("GPU", "0"))
#GPU =2
DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

print(f"[trainer] connecting to controller {CTRL_HOST}:{CTRL_PORT}, device={DEVICE}")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((CTRL_HOST, CTRL_PORT))
s.sendall(b"ROLE trainer\n")
print("[trainer] connected to controller")

#weights = torch.ones((2048, 2048), device=DEVICE)
from transformers import GPT2Model

print("[trainer] loading GPT-2 model on", DEVICE)
model = GPT2Model.from_pretrained("gpt2").to(DEVICE)
weights = torch.cat([p.flatten() for p in model.parameters()])  # flatten all params into one big tensor
print(f"[trainer] total parameters: {weights.numel()}")

step = 0
t0 = time.time()
try:
    while True:
        weights.add_(0.01)
        send_tensor(s, weights)
        step += 1
        if step % 10 == 0:
            dt = time.time() - t0
            print(f"[trainer] step={step:04d} sum={weights.sum().item():.2f} rate={step/dt:.1f} upd/s")
        time.sleep(0.2)
except (BrokenPipeError, KeyboardInterrupt):
    print("[trainer] done")
finally:
    s.close()

