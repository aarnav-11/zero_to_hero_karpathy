import torch

x1 = torch.Tensor([2]).double() ; x1.requires_grad = True
x2 = torch.Tensor([0]).double() ; x2.requires_grad = True
w1 = torch.Tensor([-3]).double() ; w1.requires_grad = True
w2 = torch.Tensor([1]).double() ; w2.requires_grad = True
b = torch.Tensor([6.88137]); b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

o.backward()

print(x2.grad.item())