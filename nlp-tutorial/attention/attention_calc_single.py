import torch
import math


n = 8
d = 512
h = 8
dk = d // 8
scale_factor = dk**(1/2)

X = torch.rand(n, d)
output_list= []

for i in range(8):
    Wk = torch.rand(d,dk)
    Wq = torch.rand(d,dk)
    Wv = torch.rand(d,dk)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    attention = Q @ K.T
    attention /= scale_factor

    causal_mask = torch.triu(torch.full((n,n), float("-inf")),diagonal=1)
    attention += causal_mask

    softmaxed = torch.softmax(attention, -1)
    output = softmaxed @ V
    output_list.append(output)

final = torch.cat(output_list, 1)

