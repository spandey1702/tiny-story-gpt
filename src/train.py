import argparse
import json
import math
import time

import torch

from device import get_device, gpu_memory_mb
from evaluate import full_val_loss
from load_data import load_data_from_files, get_batch
from model import BigramModel, TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["transformer", "bigram"], default="transformer")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--ctx", type=int, default=None, help="context length (default 64; 8 for bigram)")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=None, help="peak learning rate (default 1e-3; 1e-2 for bigram)")
parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
parser.add_argument("--warmup", type=int, default=200, help="warmup steps (cosine only)")
parser.add_argument("--n-embd", type=int, default=128)
parser.add_argument("--n-head", type=int, default=4)
parser.add_argument("--n-layer", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--train-file", default="data/train.txt")
parser.add_argument("--name", default=None, help="run name (default built from the settings)")
args = parser.parse_args()

BLOCK_SIZE = args.ctx or (8 if args.model == "bigram" else 64)
LEARNING_RATE = args.lr or (1e-2 if args.model == "bigram" else 1e-3)
RUN_NAME = args.name or f"{args.model}_{args.steps}_ctx{BLOCK_SIZE}"
EVAL_EVERY = 300
EVAL_BATCHES = 100


def lr_at(step):
    """Constant, or: rise linearly over the warmup, then follow a cosine curve
    down to 10% of the peak by the last step."""
    if args.lr_schedule == "constant":
        return LEARNING_RATE
    if step < args.warmup:
        return LEARNING_RATE * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


torch.manual_seed(1337)
device = get_device()

train_data, val_data, stoi, itos, vocab_size = load_data_from_files(args.train_file, "data/valid.txt")
print(f"device: {device} | vocabulary size: {vocab_size} | training chars: {len(train_data):,}")

if args.model == "bigram":
    model = BigramModel(vocab_size).to(device)
else:
    model = TransformerModel(vocab_size, BLOCK_SIZE, n_embd=args.n_embd, n_head=args.n_head,
                             n_layer=args.n_layer, dropout=args.dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
n_params = sum(p.numel() for p in model.parameters())
print(f"run: {RUN_NAME} | parameters: {n_params:,} | lr schedule: {args.lr_schedule}")


@torch.no_grad()
def estimate_loss():
    """Quick estimate on random batches, to follow progress during training."""
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data, BLOCK_SIZE, args.batch_size)
            _, loss = model(x.to(device), y.to(device))
            total += loss.item()
        losses[name] = total / EVAL_BATCHES
    model.train()
    return losses


history = []
start_time = time.time()
for step in range(args.steps + 1):
    lr = lr_at(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        gpu_mb = gpu_memory_mb(device)
        history.append({"step": step, "train": losses["train"], "val": losses["val"],
                        "lr": lr, "gpu_mb": gpu_mb})
        print(
            f"step {step:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
            f" | lr {lr:.1e} | GPU memory {gpu_mb:.0f} MB | {time.time() - start_time:.0f}s"
        )

    x, y = get_batch(train_data, BLOCK_SIZE, args.batch_size)
    _, loss = model(x.to(device), y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

train_seconds = time.time() - start_time

model.eval()
full_val = full_val_loss(model, stoi, device) if args.model == "transformer" else None
if full_val is not None:
    print(f"\nfull validation loss: {full_val:.4f}")

start = torch.tensor([[stoi["\n"]]], device=device)
sample = model.generate(start, max_new_tokens=300)[0].tolist()
print("\n--- sample ---")
print("".join(itos[i] for i in sample))

result = {
    "run": RUN_NAME,
    "config": vars(args) | {"ctx": BLOCK_SIZE, "lr": LEARNING_RATE},
    "device": device,
    "parameters": n_params,
    "train_seconds": round(train_seconds, 1),
    "final_val_loss": history[-1]["val"],
    "full_val_loss": full_val,
    "history": history,
    "sample": "".join(itos[i] for i in sample),
}
with open(f"results/{RUN_NAME}.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results/{RUN_NAME}.json")

torch.save(
    {"model": args.model, "block_size": BLOCK_SIZE, "n_embd": args.n_embd, "n_head": args.n_head,
     "n_layer": args.n_layer, "state_dict": model.state_dict(), "stoi": stoi, "itos": itos},
    f"checkpoints/{RUN_NAME}.pt",
)
print(f"Saved checkpoints/{RUN_NAME}.pt")
