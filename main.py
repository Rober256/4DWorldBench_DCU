import os
import time
import torch

print("=" * 60)
print("torch version:", torch.__version__)
print("torch hip:", getattr(torch.version, "hip", None))
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("HIP_VISIBLE_DEVICES:", os.environ.get("HIP_VISIBLE_DEVICES"))
print("ROCR_VISIBLE_DEVICES:", os.environ.get("ROCR_VISIBLE_DEVICES"))
print("=" * 60)

if not torch.cuda.is_available():
    raise SystemExit("PyTorch 没有检测到 DCU/GPU，请检查 DTK 环境和 torch 版本。")

for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    print(f"device {i}: {prop.name}, memory={prop.total_memory / 1024**3:.2f} GB")

device = "cuda:0"
print("using:", torch.cuda.get_device_name(0))

n = 4096
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)

torch.cuda.synchronize()
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
t1 = time.time()

print("matmul finished")
print("time:", t1 - t0)
print("checksum:", c.sum().item())
print("allocated memory GB:", torch.cuda.memory_allocated() / 1024**3)
