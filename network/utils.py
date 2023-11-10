import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            output_attn=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.output_attn = output_attn

    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))

        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.output_attn:
            return x, attn
        else:
            return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, attn_dropout=0.1, hidden_size=None, n_head=None, gating=False):
        super().__init__()
        self.temperature = hidden_size ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.gating = gating
        if self.gating:
            self.gating_bias = nn.parameter.Parameter(data=torch.ones((hidden_size*n_head,)))
            self.gating_linear = nn.Linear(hidden_size, n_head * hidden_size)

    def forward(self, q, k, v, mask=None, g=None):
        if self.gating:
            g_avg = g.mean(1)
            gate_values = self.gating_linear(g_avg)

            bs, n_head, q_len, q_dim = q.shape
            q = q.transpose(1, 2).contiguous().view(bs, q_len, -1)
            k = k.transpose(1, 2).contiguous().view(bs, q_len, -1)
            q = (1 + torch.unsqueeze(torch.sigmoid(gate_values + self.gating_bias), 1)) * q
            k = (1 + torch.unsqueeze(torch.sigmoid(gate_values + self.gating_bias), 1)) * k
            q = q.view(bs, q_len, -1, q_dim).permute(0, 2, 1, 3)
            k = k.view(bs, q_len, -1, q_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, input_size, hidden_size, dropout=0.1, output_attn=False, gating=False):
        """
            n_head: head_num,
            input_size: input_dim,

        """
        super().__init__()

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.gating = gating

        self.w_qs = nn.Linear(input_size, n_head * hidden_size)
        self.w_ks = nn.Linear(input_size, n_head * hidden_size)
        self.w_vs = nn.Linear(input_size, n_head * hidden_size)
        self.fc = nn.Linear(n_head * hidden_size, input_size)

        self.attention = ScaledDotProductAttention(attn_dropout=dropout, hidden_size=hidden_size, n_head=n_head, gating=gating)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.output_attn = output_attn

    def forward(self, q, k, v, mask=None):
        hidden_size, n_head = self.hidden_size, self.n_head
        batch_size, length = v.shape[0], v.shape[1]

        residual = q
        g = q

        q = self.w_qs(q).view(batch_size, length, n_head, hidden_size)
        k = self.w_ks(k).view(batch_size, length, n_head, hidden_size)
        v = self.w_vs(v).view(batch_size, length, n_head, hidden_size)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask, g=g)

        q = q.transpose(1, 2).contiguous().view(batch_size, length, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        if self.output_attn:
            return q, attn
        else:
            return q
