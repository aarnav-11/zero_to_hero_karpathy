
#setting up SHA training

import torch
import math
import torch.nn as nn

n = 2048
d = 8
h = 1
dh = d // h
lr = 0.01

Wq = torch.randn(d,dh, requires_grad = True)
Wk = torch.randn(d,dh, requires_grad = True)
Wv = torch.randn(d,dh, requires_grad = True)


def norm(x):
    return x #for now

def tokenize(n,d):
    return torch.randn(n, d) #for now; in actual we have tokenizer

X = norm(tokenize(n,d))

def forward():
    Q = norm(X @ Wq)
    K = norm(X @ Wk)
    V = X @ Wv

    inside = Q @ K.T
    inside = inside / (pow(dh, 0.5))
    causal_mask = torch.triu(torch.full((n,n), float("-inf")), diagonal=1)
    inside = inside + causal_mask
    attention = torch.softmax(inside, dim=1)

    output = attention@V
    return output