import torch
# with open('TinyStoriesV2-GPT4-valid.txt', 'r', encoding='utf-8') as f:
#         text = f.read()

# chars = sorted(list(set(text)))
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for ch, i in stoi.items()}
# def encode(s: str):
#     return [stoi[c] for c in s]

# def decode(ids):
#     return "".join([itos[i] for i in ids])

# data = torch.tensor(encode(text), dtype=torch.long)
# print(max(data))
# n = int(0.9*len(data))
# train_data = data[:n]
# val_data = data[n:]
# vocab_size = len(chars)

# def test_get_batch():
#     data = torch.arange(20)
#     block_size = 5
#     batch_size = 3


#     ix = torch.randint(len(data) - block_size - 1, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])

#     print("data:", data)
#     print("ix:", ix)
#     print("x shape:", x.shape)
#     print("y shape:", y.shape)
#     print("x:")
#     print(x)
#     print("y")

# test_get_batch()

x = torch.randint(10, (2,10,5))
y = x.view(-1, x.size(-1))
print(x)
print(y)
