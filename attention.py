import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b, attn=None, Wattn=None):
    N, H = prev_h.shape
    if attn is None:
        vanilla = torch.matmul(prev_h, Wh) + torch.matmul(x, Wx) + b
    else:
        vanilla = torch.matmul(prev_h, Wh) + torch.matmul(x, Wx) + torch.matmul(attn, Wattn) + b
    i = torch.sigmoid(vanilla[:, 0:H])
    f = torch.sigmoid(vanilla[:, H:(2 * H)])
    o = torch.sigmoid(vanilla[:, (2 * H):(3 * H)])
    g = torch.tanh(vanilla[:, (3 * H):(4 * H)])
    next_c = f * prev_c + i * g
    next_h = o * torch.tanh(next_c)
    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    c0 = torch.zeros_like(h0)  # we provide the intial cell state c0 here for you!
    N, T, D = x.shape
    N, H = h0.shape
    h = torch.zeros(size=(N, T, H), device=x.device, dtype=x.dtype)
    prev_h = h0
    prev_c = c0
    for t in range(T):
        prev_h, prev_c = lstm_step_forward(x=x[:, t, :], prev_h=prev_h, prev_c=prev_c, Wx=Wx, Wh=Wh, b=b)
        h[:, t, :] = prev_h
    return h


def attention_forward(x, A, Wx, Wh, Wattn, b):
    h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
    c0 = h0  # Initial cell state, of shape (N, H)

    N, T, D = x.shape
    N, H = h0.shape
    h = torch.zeros(size=(N, T, H), device=x.device, dtype=x.dtype)
    prev_h = h0
    prev_c = c0
    for t in range(T):
        attn, attn_weights = dot_product_attention(prev_h, A)
        prev_h, prev_c = lstm_step_forward(x=x[:, t, :], prev_h=prev_h, prev_c=prev_c, Wx=Wx, Wh=Wh, b=b, attn=attn,
                                           Wattn=Wattn)
        h[:, t, :] = prev_h
    return h


def dot_product_attention(prev_h, A):
    N, H, D_a, _ = A.shape
    A = A.reshape(N, H, 16)
    prev_h = prev_h.reshape(N, 1, H)
    M = torch.matmul(prev_h, A) / (H ** 0.5)
    M = torch.nn.functional.softmax(M, dim=2)
    attn = torch.matmul(A, M.reshape(N, 16, 1)).reshape(N, H)
    attn_weights = M.reshape(N, 4, 4)
    return attn, attn_weights


class AttentionLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, device='cpu',
                 dtype=torch.float32):
        super().__init__()

        # Register parameters
        self.Wx = Parameter(torch.randn(input_size, hidden_size * 4,
                                        device=device, dtype=dtype).div(math.sqrt(input_size)))
        self.Wh = Parameter(torch.randn(hidden_size, hidden_size * 4,
                                        device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.Wattn = Parameter(torch.randn(hidden_size, hidden_size * 4,
                                           device=device, dtype=dtype).div(math.sqrt(hidden_size)))
        self.b = Parameter(torch.zeros(hidden_size * 4,
                                       device=device, dtype=dtype))

    def forward(self, x, A):
        hn = attention_forward(x, A, self.Wx, self.Wh, self.Wattn, self.b)
        return hn

    def step_forward(self, x, prev_h, prev_c, attn):
        next_h, next_c = lstm_step_forward(x, prev_h, prev_c, self.Wx, self.Wh,
                                           self.b, attn=attn, Wattn=self.Wattn)
        return next_h, next_c
