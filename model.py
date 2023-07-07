import torch.nn as nn
import torch 
import math
import os 

class TransformerConfig():
    def __init__(self, vocab_size = 104, emb_dim = 384, seq_len=256, n_heads=6, n_layers=8):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers

class TransformerModel:
    def __init__(self, device):
        config = TransformerConfig()
        self.transformer = Transformer(config.vocab_size, config.emb_dim, config.seq_len, config.n_heads, config.n_layers)
        state_dict = torch.hub.load_state_dict_from_url(f"https://huggingface.co/kir486680/matsci-model/resolve/main/model_causal_materials.pth", map_location=device)
        self.transformer.load_state_dict(state_dict['model_state_dict'])
        self.transformer.to(device)  
    
        
    def forward(self, input_ids):
        return self.transformer(input_ids)



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight



class CausalSelfAttention(nn.Module):

    def __init__(self, emb_dim, n_heads):
        super().__init__()
        assert emb_dim % n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim)
        # output projection
        self.c_proj = nn.Linear(emb_dim, emb_dim)
        # regularization
        self.attn_dropout = nn.Dropout(0.2)
        self.resid_dropout = nn.Dropout(0.2)
        self.n_head = n_heads
        self.n_embd = emb_dim
        #support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(context_length, context_length))
                                        .view(1, 1, context_length, context_length))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.2 if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super(TransformerBlock, self).__init__()
        head_size = emb_dim // n_heads
        self.multihead = CausalSelfAttention(emb_dim, n_heads)
        self.layer_norm1 = RMSNorm(emb_dim)
        self.layer_norm2 = RMSNorm(emb_dim)
        self.ffd = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(0.2)
        ) 
        
    def forward(self, input):
        # Multi-head Attention Sublayer
        x = input + self.multihead(self.layer_norm1(input))
        x = x + self.ffd(self.layer_norm2(x))

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, n_heads, n_layers):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, seq_len)
        self.layers = nn.ModuleList([TransformerBlock(emb_dim, n_heads) for _ in range(n_layers)])
        self.ffd = nn.Linear(emb_dim, vocab_size)

    def forward(self, input):
        token_embedded = self.token_embedding(input)
        embedded = self.positional_encoding(token_embedded)
        for layer in self.layers:
            embedded = layer(embedded)
        logits = self.ffd(embedded)
        return logits