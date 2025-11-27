import socket, struct, io, torch

def send_tensor(sock: socket.socket, t: torch.Tensor):
    meta = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}
    buf = io.BytesIO(); torch.save(meta, buf); meta_bytes = buf.getvalue()
    data = t.detach().cpu().contiguous().numpy().tobytes()
    sock.sendall(struct.pack("!Q", len(meta_bytes))); sock.sendall(meta_bytes)
    sock.sendall(struct.pack("!Q", len(data)));       sock.sendall(data)

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk: raise ConnectionError("socket closed")
        out += chunk
    return bytes(out)

def recv_tensor(sock: socket.socket, device=None) -> torch.Tensor:
    meta_len = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    meta = torch.load(io.BytesIO(_recv_exact(sock, meta_len)))
    data_len = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    raw = _recv_exact(sock, data_len)

    dtype = getattr(torch, meta["dtype"])
    t = torch.frombuffer(memoryview(raw), dtype=dtype).reshape(meta["shape"]).clone()
    if device is not None: t = t.to(device, non_blocking=True)
    return t

