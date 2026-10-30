import json
import sys
import time

import torch

from load_data import load_data_from_files, get_batch
from model import BigramModel, TransformerModel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "transformer"  # "bigram" or "transformer"

BATCH_SIZE = 32
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
EVAL_EVERY = 300
EVAL_BATCHES = 100

if MODEL == "bigram":
    BLOCK_SIZE = 8
    LEARNING_RATE = 1e-2
else:
    BLOCK_SIZE = 64
    LEARNING_RATE = 1e-3

RUN_NAME = f"{MODEL}_{STEPS}"

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"

train_data, val_data, stoi, itos, vocab_size = load_data_from_files(
    "data/train.txt",
    "data/valid.txt",
)
print("device:", device, "| vocabulary size:", vocab_size)

if MODEL == "bigram":
    model = BigramModel(vocab_size).to(device)
else:
    model = TransformerModel(vocab_size, BLOCK_SIZE).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            _, loss = model(x.to(device), y.to(device))
            total += loss.item()
        losses[name] = total / EVAL_BATCHES
    model.train()
    return losses


history = []
start_time = time.time()
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        gpu_mb = torch.mps.driver_allocated_memory() / 1e6 if device == "mps" else 0.0
        history.append({"step": step, "train": losses["train"], "val": losses["val"], "gpu_mb": gpu_mb})
        print(
            f"step {step:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}"
            f" | GPU memory {gpu_mb:.0f} MB | {time.time() - start_time:.0f}s"
        )

    x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
    _, loss = model(x.to(device), y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

train_seconds = time.time() - start_time

start = torch.tensor([[stoi["\n"]]], device=device)
sample = model.generate(start, max_new_tokens=300)[0].tolist()
print("\n--- sample ---")
print("".join(itos[i] for i in sample))

result = {
    "run": RUN_NAME,
    "parameters": sum(p.numel() for p in model.parameters()),
    "train_seconds": round(train_seconds, 1),
    "final_val_loss": history[-1]["val"],
    "history": history,
    "sample": "".join(itos[i] for i in sample),
}
with open(f"results/{RUN_NAME}.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved results/{RUN_NAME}.json")

torch.save(
    {"model": MODEL, "block_size": BLOCK_SIZE, "state_dict": model.state_dict(), "stoi": stoi, "itos": itos},
    f"checkpoints/{RUN_NAME}.pt",
)
print(f"Saved checkpoints/{RUN_NAME}.pt")
