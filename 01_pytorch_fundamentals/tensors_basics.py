
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.randn(3)

print("x:", x)
print("y:", y)

z = x + y
print("x + y:", z)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

print("Using device:", device)
