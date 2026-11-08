
import torch
from torch.utils.benchmark import Timer

from pathlib import Path
from torch.utils.cpp_extension import load

file_path = Path(__file__).resolve().parent
print(file_path)

