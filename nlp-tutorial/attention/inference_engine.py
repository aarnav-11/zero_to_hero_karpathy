
import torch

n = 256
d = 2048
h = 16
dh = d // h

X = torch.rand(n,d)

Wk = torch.rand(d,dh)
Wq = torch.rand(d,dh)
Wv = torch.rand(d,dh)

def attention(Wk, Wq, Wv, X):
    final_mat = []
    for _ in range(h):
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        attntn = Q @ K.T
        attntn = attntn / pow(dh, 1/2)
        causalMask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
        attntn = attntn + causalMask
        softmaxed = torch.softmax(attntn, -1)

        final = softmaxed @ V
        final_mat.append(final)
    output = torch.cat(final_mat, 1)
    return output

hidden_dim = 4*d

def FFN(input):
    # relu(W1+b1)*W2 + b
    W1 = torch.random(d, hidden_dim) #d,hidden
    W2 = torch.random(hidden_dim, d)

    b1 = torch.ones(hidden_dim)
    b2 = torch.zeros(d)

    inside = input @ W1 + b1
    relu = torch.relu(inside)
    final = relu @ W2 + b2

    return final

def single_layer(input):
    a = attention(Wk, Wq, Wv, input)
    b = FFN(a)
    return b


