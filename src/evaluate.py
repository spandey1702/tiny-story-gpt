import sys

import torch

from device import get_device
from generate import load_model


@torch.no_grad()
def full_val_loss(model, stoi, device, batch=64):
    """Loss over the whole validation set in back-to-back windows: identical every run.
    The text is encoded with the model's own vocabulary, so models trained on
    different data are still scored on exactly the same characters."""
    with open("data/valid.txt", encoding="utf-8") as f:
        val = torch.tensor([stoi[c] for c in f.read()], dtype=torch.long)
    T = model.block_size
    n = (len(val) - 1) // T * T
    x, y = val[:n].view(-1, T), val[1 : n + 1].view(-1, T)
    total = 0.0
    for i in range(0, len(x), batch):
        _, loss = model(x[i : i + batch].to(device), y[i : i + batch].to(device))
        total += loss.item() * x[i : i + batch].numel()
    return total / n


if __name__ == "__main__":
    runs = sys.argv[1:] or ["transformer_10000_ctx32", "transformer_10000",
                            "transformer_10000_ctx128", "transformer_10000_ctx256"]
    device = get_device()
    for run in runs:
        model, stoi, _ = load_model(run, device)
        print(f"{run:28} full val loss {full_val_loss(model, stoi, device):.4f}")
