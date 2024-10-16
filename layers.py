import torch
import torch.nn as nn
import math
from utils import clones
import torch.nn.functional as F
from torch.nn.functional import log_softmax


class LayerNorm(nn.Module):
    "Construct a layernorm module - https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection (https://arxiv.org/abs/1512.03385) followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # Dimension of key/query vectors
    # Perform scaled dot-product attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply the mask if present
    if mask is not None:
        if mask.dim() == 2:  # If 2D (batch, seq_len), add an extra dimension
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)  # Mask padded values

    # Softmax and attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout, if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Final matrix multiplication with the value vector
    output = torch.matmul(attention_weights, value)
    return output, attention_weights



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_v=None, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # Ensure that d_model is divisible by the number of heads

        # d_k is the dimension for key and query vectors, d_v for value vectors
        self.d_k = d_model // h
        self.d_v = d_v if d_v else self.d_k  # If d_v is not provided, default to d_k
        self.h = h
        
        # Define the projection layers for Q, K, V and the final linear layer
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, self.d_v * h)  # Use d_v for value projection
        self.linear = nn.Linear(self.d_v * h, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, attn_mask=None):
        batch_size = Q.size(0)

        # Project Q, K, V and reshape for multi-head attention
        q_s = self.W_Q(Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)  # Project values to d_v

        # Apply scaled dot-product attention
        context, attention_weights = attention(q_s, k_s, v_s, attn_mask)

        # Save the attention weights for later use
        self.attn = attention_weights

        # Concatenate heads and apply the final linear projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)
        output = self.linear(context)
        
        return output, attention_weights


    
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

    

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())    

    
