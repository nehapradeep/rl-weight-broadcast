# test_uccl_collective.py
import torch
torch.cuda.set_device(0)

from uccl import collective

print("Initializing UCCL collective...")
collective.init_collective(num_cpus=4)

print("SUCCESS! UCCL collective initialized")

# Create test tensor
tensor = torch.randn(1000, device='cuda:0')
collective.register_tensor(tensor)

print(f"Tensor registered: shape={tensor.shape}")

collective.finalize_collective()
print("UCCL collective finalized")