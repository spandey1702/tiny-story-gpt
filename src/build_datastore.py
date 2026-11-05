import sys
import time

import torch

from device import get_device
from generate import load_model
from load_data import load_data_from_files

RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "transformer_10000_ctx256"
BATCH = 64      # windows per forward pass

device = get_device()
model, stoi, itos = load_model(RUN_NAME, device)
train_data, _, _, _, _ = load_data_from_files("data/train.txt", "data/valid.txt")

# Cut the training text into back-to-back windows the model can read
T = model.block_size
n = (len(train_data) - 1) // T * T
x = train_data[:n].view(-1, T)            # inputs
y = train_data[1 : n + 1].view(-1, T)     # the character that came next

dim = model.ln_final.normalized_shape[0]
keys = torch.empty(n, dim, dtype=torch.float16)   # float16 halves memory
start = time.time()
with torch.no_grad():
    for i in range(0, len(x), BATCH):
        feats = model.features(x[i : i + BATCH].to(device))
        keys[i * T : i * T + feats.numel() // dim] = feats.reshape(-1, dim).half().cpu()

values = y.reshape(-1)
out_path = f"checkpoints/datastore_{RUN_NAME}.pt"
torch.save({"keys": keys, "values": values}, out_path)
print(f"{n:,} entries ({keys.numel() * 2 / 1e6:.0f} MB) in {time.time() - start:.0f}s → {out_path}")
