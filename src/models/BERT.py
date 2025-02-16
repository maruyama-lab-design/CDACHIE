import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.qkv_linear = nn.Linear(d_model, d_model*3, bias = False)
        self.out_linear = nn.Linear(d_model, d_model, bias = False)

    def forward(self, x):
        x = self.norm(x)                                                                                    # x: [batch, seq_len, d_model]
        qW, kW, vW = self.qkv_linear(x).chunk(3, dim = -1)                                                  # qW, kW, vW: [batch, seq_len, d_model]
        qW, kW, vW = map(lambda x: rearrange(x, 'b s (h d) -> b h s d', h = self.num_heads), (qW, kW, vW))  # qW, kW, vW: [batch, num_heads, seq_len, dim_head]

        qk = torch.einsum('b h i d, b h j d -> b h i j', qW, kW)                                            # qk: [batch, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(torch.softmax(qk/(self.dim_head ** 0.5), dim=-1))                       # attn_weights: [batch, num_heads, seq_len, seq_len]

        out = torch.einsum('b h i j, b h j d -> b h i d', attn_weights, vW)                                 # out: [batch, num_heads, seq_len, dim_head]
        out = rearrange(out, 'b h s d -> b s (h d)', h = self.num_heads)                                    # out: [batch, seq_len, d_model]
        out = self.out_linear(out)                                                                          # out: [batch, seq_len, d_model]

        return out, attn_weights

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(d_model, dropout=0.1, dim_feedforward=128):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, dim_feedforward*2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model)
    )

class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=128, dropout=0.5, num_layers=6,
                  share_attn=False, share_ffn=False):
        super().__init__()
        self.attn = MultiheadAttention(d_model, num_heads, dropout) if share_attn else None
        self.ffn = FeedForward(d_model, dropout, dim_feedforward) if share_ffn else None
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                self.attn if share_attn else MultiheadAttention(d_model, num_heads, dropout),
                self.ffn if share_ffn else FeedForward(d_model, dropout, dim_feedforward)
                ]))
            
    def forward(self, x):
        attn_weights_list = []
        for attn, ffn in self.layers:
            attn_out, attn_weights = attn(x)
            attn_weights_list.append(attn_weights)
            x = x + attn_out
            x = x + ffn(x) 
        return x, torch.stack(attn_weights_list)
    

class Embedder(nn.Module):
    def __init__(self, input_size, d_input, d_model):
        super().__init__()
        if d_model % d_input != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by d_input ({d_input})')
        self.weights = nn.Parameter(torch.randn(input_size, d_model))
        self.biases = nn.Parameter(torch.randn(input_size, d_model))

        self.d_input = d_input
        self.d_model = d_model

    def forward(self, x): # x: [batch, seq_len, d_input]
        x = repeat(x, 'b n d -> b n (d c)', c=self.d_model//self.d_input) if self.d_input >= 2 else x
        return x * self.weights + self.biases # out: [batch, seq_len, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0,1))

    def forward(self, x): # x: [batch, seq_len, d_model]
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x) # out: [batch, seq_len, d_model]



# BERT models

# class BERT100kb(nn.Module):
#     def __init__(self, d_model, num_layers, num_heads, dropout, dim_feedforward, input_size, output_size, d_input=1, share_attn=False, share_ffn=False, cls_init='random'):
#         super().__init__()
#         self.embedder = Embedder(input_size, d_input, d_model)
#         if cls_init == 'random':
#             self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
#         elif cls_init == 'zeros':
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
#         elif cls_init == 'ones':
#             self.cls_token = nn.Parameter(torch.ones(1, 1, d_model))
#         else:
#             raise ValueError(f'cls_init must be "random", "zero" or "ones", but got {cls_init}')
#         self.transformer_encoder = CustomTransformerEncoder(d_model, num_heads, dim_feedforward, dropout, num_layers, share_attn, share_ffn)
#         self.to_logits = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, output_size)
#         )

#     def forward(self, x): # x: [batch_size, seq_length]
#         x = torch.unsqueeze(x, dim=-1)
#         x = self.embedder(x)
#         x = torch.cat((self.cls_token.repeat(x.shape[0], 1, 1), x), dim=1)
#         x, attn_weights = self.transformer_encoder(x)
#         cls = x[:, 0, :]
#         logits = self.to_logits(cls)

#         return logits, attn_weights, cls # logits: [batch_size, output_size]

class BERT1kb(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dropout, dim_feedforward, input_size, output_size, d_input=12, share_attn=False, share_ffn=False, cls_init='random'):
        super().__init__()
        self.linear = nn.Linear(d_input, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        if cls_init == 'random':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        elif cls_init == 'zeros':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        elif cls_init == 'ones':
            self.cls_token = nn.Parameter(torch.ones(1, 1, d_model))
        self.transformer_encoder = CustomTransformerEncoder(d_model, num_heads, dim_feedforward, dropout, num_layers, share_attn, share_ffn)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_size)
        )

    def forward(self, x): # x: [batch_size, seq_length, d_input]
        x = self.linear(x)
        x = self.positional_encoding(x)
        x = torch.cat((self.cls_token.repeat(x.shape[0], 1, 1), x), dim=1)
        x, attn_weights = self.transformer_encoder(x)
        cls = x[:, 0, :]
        logits = self.to_logits(cls)

        return logits, attn_weights, cls # logits: [batch_size, output_size]

    



