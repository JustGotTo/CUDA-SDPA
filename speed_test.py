
import torch
from torch.utils.benchmark import Timer

from pathlib import Path
from torch.utils.cpp_extension import load

file_path = Path(__file__).resolve().parent

attention_cuda = load(
    name="cuda_attention",
    sources=[str(file_path / "cuda_sdpa.cu")],
    verbose=True,
    extra_cuda_cflags=["-O2"]
)

#Creating Test Environment
torch.manual_seed(42)

ndim = 128
D = 64
Dv = 64

Q = torch.randn(
    ndim, D,
    device="cuda",
    dtype=torch.float32
)

K = torch.randn(
    ndim, D,
    device="cuda",
    dtype=torch.float32
)

V = torch.randn(
    ndim, Dv,
    device="cuda",
    dtype=torch.float32
)

start = Timer.timer()
attention_result = attention_cuda.attention(Q, K, V)
end = Timer.timer() - start

print(f"Time taken: {end} seconds")

#Comparison to pytorch

start = Timer.timer()
attn_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
end = Timer.timer() - start

print(f"Time taken: {end} seconds")



