import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Config
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337

    # data
    data_path: str = "TinyStoriesV2-GPT4-valid.txt"
    block_size: int = 128 # context length 最大上下文长度
    batch_size: int = 64

    # model
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1

    # train
    lr: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 100

    # generation
    gen_len: int = 300
    temperature: float = 1.0
    top_k: int = 50

cfg = CFG()

# utils
def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def estimate_loss(model, get_batch):
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        total = 0.0
        for _ in range(cfg.eval_iters):
            xb, yb = get_batch(split)
            _, loss, _ = model(xb, yb)
            total += loss.item()
        losses[split] = total / cfg.eval_iters
    model.train()
    return losses

# Data(char-level)

#实现功能：
# 1. read file 得到text
# 2. text -> chars -> stoi and itos
# 3. vocab_size = len(chars)
# 4. en
def load_data(path:str):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, encode, decode


# arg:train/val data
# return: batches stack

def make_get_batch(train_data, val_data, cfg):
    def get_batch(split:str):
        """
        从一条很长的 token 序列中，
        随机抽取 B 段长度为 T 的连续子序列，
        并让模型在每一个位置
        学习预测下一个 token
        data: (N,) 整个文本的token序列
        ix: (B,) 每个batch的其实位置
        x: (B,T) 输入token
        y: (B, T) 目标token（右移一位）

        example:
        i = 2 block_size=4
        x = data[2:6] = [2, 3, 4, 5]
        y = data[3:7] = [3, 4, 5, 6]
        P(3|2)
        P(4|23)
        P(5|234)
        P(6|2345)
        Transformer 并行预测每一个位置的下一个 token
        """
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,)) # randint(high, size) 生成 [0,high)范围，形状(batch_size,)
        x = torch.stack([data[i:i+cfg.block_size] for i in ix])  #一共batch_size个 block_size长度的tensor data, stack 就是把这些tensor在新维度上拼起来，每个tensor(T,),拼起来就是(B,T)
        y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])
        return x.to(cfg.device), y.to(cfg.device)
    return get_batch

# Model components

# Multi-head Self Attention
class CausalSelfAttention(nn.Module):
    """
    B batch size 一次并行多少段文本
    T 序列长度 block_size 当前上下文长度
    C n_embd embedding维度
    nh head数 
    hs head维度 
    """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3*n_embd, bias=False) #(B,T,3C)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

        # causal mask (buffer so it moves with device, not trained)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x, past_kv=None):
        """
+        x: (B, T, C)
+        past_kv: None or tuple (k_past, v_past) each shape (B, nh, T_past, hs)
+        returns: y (B, T, C), present_kv (k_all, v_all)
+        """
        B, T, C = x.shape

        qkv = self.qkv(x) # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2) #把dim=2即3C那个维度 切成每段C长度的 

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) #只是重排内存视图，不是复制数据 拆成（B, T, nh, hs）
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # transpose(1,2)意思是交换第一维和第二维 (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # 所以每个qkv都是相同的形状 后续attn计算在每个head内做T*T的相关性矩阵

        """
        基础attn
        # scaled dot-product attention
        # q 是 (B, n_head, T, head_dim)
        # k^T 是 (B, n_head, head_dim, T)
        # att (B, n_head, T, T) 对每个batch 每个head 得到一个T*T的相似度矩阵：
        # att[..., t, s]表示 第t个token的query与第s个token的key的相似度（打分）
        """
        # att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, nh, T, T)
        
        # 如果不除，head_dim 越大，点积的数值范围越大，softmax 会饱和（接近 0/1），梯度变小训练不稳定。
        # 缩放保证数值稳定，这是标准做法

        # 对于未来位置(s>t) 把分数置为-inf after softmax 这些位置的prob=0
        # 保证自回归： 第t个token只能看<=t的token
        
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        """
        加了KV cache版本
        """
        if past_kv is not None:
            k_past, v_past = past_kv  # each (B, nh, T_past, hs)
            # concatenate on time axis (dim=2)
            k = torch.cat([k_past, k], dim=2)  # (B, nh, T_all, hs)
            v = torch.cat([v_past, v], dim=2)
        
        att = ( q @ k.transpose(-2, -1) / math.sqrt(self.head_dim) )
        # 拼接后的T_all长度有可能超过T, 用slice mask
        T_all = k.shape[2]
        att = att.masked_fill(self.mask[:, :, :T, :T_all] == 0, float("-inf"))

        # dim=-1最后一维，对每个t, 在s=0,...T-1上归一化：
        # 每一行（固定t) 变成概率分布， 表示当前token应该关注哪些历史token
        # shape不变
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        # y=att@v 用权重对value做加权和 对历史信息的加权汇聚
        # y_t ​= sum_{s≤t} ​α_{t,s} * ​v_s​
        y = att @ v #（T,T) @ (T, head_dim) -> (T, head_dim)
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C) 把同一token的多个head放在一起
        y = y.transpose(1,2).contiguous().view(B, T, C)
        # self.proj = nn.Linear(C, C)
        # 多头拼接只把信息堆起来，proj负责把各个head的信息重新混合(learned mixing)
        y = self.drop(self.proj(y))

        # present_kv: 返回能被caller cache的full k,v 
        # detach 以避免holding computation graph during generation
        present_kv = (k.detach(), v.detach())
        return y, present_kv
    
# MLP 
class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 多层Transformer Block: 每层=LN + Attn + LN+MLP + 残差
# x <- x+Attn(LN(x)) / x+MLP(LN(x)), 最后linear head输出vocab logits
# 1. 为什么要残差，避免梯度消失，让每层学会增量修正，不是重写表示
# 2. 原始transformer bert post LN; GPT pre LN 现代大模型 Pre-LN/RMSNorm变体
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, past=None):
        # x = x + self.attn(self.ln1(x))
        # x = x + self.mlp(self.ln2(x))

        """
        past: None or tuple (k_past, v_past) for this layer
        returns: x_out, present_kv
        """
        attn_out, present = self.attn(self.ln1(x), past)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size

        # nn.Embedding(idx, dim)
        self.tok_emb = nn.Embedding(vocab_size, n_embd) # 输入token id idx, (B, T) 输出 （B, T, C) C=n_embd
        self.pos_emb = nn.Embedding(block_size, n_embd) # input: id 0...T-1 (T，) output: (T, C)
        self.drop = nn.Dropout(dropout)

        # 堆叠n_layer个transformer block
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]
        )
        # final layer norm 让输出分布更稳定（gpt常见结构）
        self.ln_f = nn.LayerNorm(n_embd)
        # 输出头， 把每个位置的hidden state映射到词表大小
        # (B, T, C) -> (B, T, V) 输出长为V的logits向量，用来预测下一个token是哪个
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # 权重共享 让输出头的权重矩阵和输入embedding的权重矩阵共享 都是(V, C)
        self.lm_head.weight = self.tok_emb.weight

    # idx: (B, T) token ids
    # targets: (B, T) next-token labels
    def forward(self, idx, targets=None, pasts=None):

        B, T = idx.shape
        # pos embedding和Attn mask buffer都是和block size预建的，所以T不能超出
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")
        tok = self.tok_emb(idx) # (B, T, C)
        # (T, C) arange(T) 生成[0,1,..,T-1] 这里没有batch维度，因为pos对batch内所有样本都一样
        pos = self.pos_emb(torch.arange(T, device=idx.device)) #(T, C)
        # broadcast广播 pos自动变粗(1, T, C) 再加到每个batch上 结果仍然是(B, T, C) 然后dropout正则化
        x = self.drop(tok + pos)

        # for blk in self.blocks:
        #     x = blk(x)
        
        # KV-cache版本：
        presents = []
        # 遍历blocks 并传入每层的past
        for i, blk in enumerate(self.blocks):
            layer_past = None if pasts is None else pasts[i]
            x, present = blk(x, past=layer_past)
            presents.append(present)

        x = self.ln_f(x)
        # logits位置t，对应的基于x[:t]的信息对下一个token的预测
        logits = self.lm_head(x) # (B, T, vocab)
        
        loss = None
        if targets is not None:
            # flatten for cross entropy
            # cross entropy 的期望输入(N, C)
            # N 样本数(B*T) C 类别数（vocab size)
            # logits_2d (B*T, V) targets_1d (B*T,)
            # log softmox
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, presents
        
    @torch.no_grad()    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        假设：
        B=2
        vocab V=100
        prompt 长度 T=5
        block_size=4
        则：
        初始 idx: (2, 5)
        crop：idx_cond: (2, 4)（只取最后 4 个）
        model 输出 logits：(2, 4, 100)
        取最后一步：(2, 100)
        softmax：(2, 100)
        multinomial：next_id: (2, 1)
        拼接后 idx：(2, 6)
        循环 300 次，最终 (2, 305)。
        """
        self.eval() # dropout关闭（避免生成随机抖动更大，不符合训练预期） LN正常工作
        B = idx.size(0)

        # 初始化empty pasts(1层一个)
        pasts = [None] * len(self.blocks)
        
        for _ in range(max_new_tokens):
            # idx (B, T_current) 初始T_current=prompt长度
            # 每生成一步， T_current += 1
            # -self.block_size: 表示min(T_current, block_size)
            # 最终idx_cond (B,min) 上下文窗口
            # idx_cond = idx[:, -self.block_size:]
            idx_cond = idx[:, -1].to(next(self.parameters()).device)
            # logits 前向传播返回的， shape (B, T_cond, V)
            # logits, _ = self(idx_cond)
            logits, _, presents = self(idx_cond, pasts=pasts)

            # present是list of [k_all, v_all] per layer
            pasts = presents

            # logits[:, -1, :] 代表：在当前位置（最后一个 token 之后）预测下一个 token 的 logits；形状变成：(B, V)
            # temperature < 1 更确定（分布更尖锐）； > 1 更发散（更随机）
            logits = logits[:, -1, :] / max(temperature, 1e-8)


            if top_k is not None:
                # v 每行最大的k个值，形状(B, K)
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                # 把logits里除了前k个之外的都设置为-inf，softmax之后就变成0了，就可以在top-k候选里采样了
                logits[logits < v[:, [-1]]] = -float("inf")
            #（B,V) -> (B,V)每行和为1
            probs = F.softmax(logits, dim=-1)
            # （B, 1) 
            next_id = torch.multinomial(probs, num_samples=1)
            # idx 从 (B, T_current) 变成 (B, T_current+1)
            idx = torch.cat([idx, next_id], dim=1)

        # idx (B, prompt_len + max_new_tokens)
        return idx


def main():
    set_seed(cfg.seed)
    train_data, val_data, vocab_size, encode, decode = load_data(cfg.data_path)
    get_batch = make_get_batch(train_data, val_data, cfg)   # 闭包：get_batch 可以直接访问 train/val

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=cfg.dropout,
    ).to(cfg.device)
    # adam+decoupled weight decay
    # model.param 把所有可训练参数交给更新
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    t0 = time.time()

    for it in range(1, cfg.max_iters + 1):
        # 训练那一套流程
        xb, yb = get_batch('train')
        # model()返回两个值： logits and loss
        # 训练阶段只需要loss
        _, loss, _ = model(xb, yb)

        # pytorch grad是累加的，每一步反向传播前必须清零
        optimizer.zero_grad(set_to_none=True)
        # 自动微分计算所有参数的grad param.grad
        loss.backward()
        # AdamW 根据梯度与动量项更新模型参数
        optimizer.step()

        if it % cfg.eval_interval == 0 or it == 1:
            # estimate_loss通常会在 torch.no_grad() 下（不计算梯度）分别对 train/val 采样若干 batch，取平均 loss
            losses = estimate_loss(model, get_batch)
            dt = time.time() - t0
            print(
                f"iter {it:4d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {dt:.1f}s"
            )
    
    # ===== save model weights =====
    save_path = "minigpt_weights.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    
    # Generation demo
    load_path = "minigpt_weights.pt"
    model.load_state_dict(torch.load(load_path, map_location=cfg.device))
    model.to(cfg.device)
    print(f"Loaded model weights from {load_path}")

    prompt = "To be"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=cfg.device)
    out = model.generate(
        idx,
        max_new_tokens=cfg.gen_len,
        temperature=cfg.temperature,
        top_k=cfg.top_k
    )[0].tolist()
    
    print("prompt")
    print(prompt)
    print("completion")
    print(decode(out))


if __name__ == "__main__":
    main()
    

