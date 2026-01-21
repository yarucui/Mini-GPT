import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Config
class CFG:
    device: str = 
    seed: int = 1337

    # data
    data_path: str = 
    block_size: int =  # context length
    batch_size: int = 

    # model
    n_embd: int = 
    n_head: int = 
    n_layer: int = 
    dropout: float = 

    # train
    epoch: int = 
    lr: float = 
    max_iters: int =
    eval_interval: int = 
    eval_iters: int = 

    # generation
    gen_len: int = 
    temperature: float = 
    top_k: int 

cfg = CFG()

# utils
def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def estimate_loss(model, get_batch):

    return losses

# Data(char-level)

#实现功能：
# 1. read file 得到text
# 2. text -> chars -> stoi and itos
# 3. vocab_size = len(chars)
# 4. en
def load_data(path:str):
    
    return train_data, val_data, vocab_size, encode, decode


# arg:train/val data
# return: batches stack
def get_batch(split:str):

    return x.to(cfg.device), y.to(cfg.device)

# Model components

# Multi-head Self Attention
class CasualSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = 

        self.qkv = 
        self.proj = 
        self.drop = 

         # causal mask (buffer so it moves with device, not trained)
    
    def forward(self, x):

        return y
    
# MLP 
class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        return x

# 多层Transformer Block: 每层=LN + Attn + LN+MLP + 残差
# x <- x+Attn(LN(x)) / x+MLP(LN(x)), 最后linear head输出vocab logits
# 1. 为什么要残差，避免梯度消失，让每层学会增量修正，不是重写表示
# 2. 原始transformer bert post LN; GPT pre LN 现代大模型 Pre-LN/RMSNorm变体
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausualSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()

    def forward(self, idx, targets=None):

        return logits, loss

@torch.no_grad()    
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):


    return idx


def main():
    set_seed(cfg.seed)

    model = MiniGPT(

    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    t0 = time.time()

    for it in range(1, cfg.max_iters + 1):
        # 训练那一套流程
    
    prompt = "To be"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=cfg.device)
    out = model.generate(...)[0].tolist()
    
    print("prompt")
    print(prompt)
    pirnt("completion")
    print(decode(out))


if __name__ = "__main__":
    main()
    

