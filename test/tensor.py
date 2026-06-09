import torch

a = torch.tensor(1)
print(a)
b = torch.tensor((1,))
print(b)
a = a.unsqueeze(0)
print(a)