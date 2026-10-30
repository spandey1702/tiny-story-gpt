import json
import textwrap

import matplotlib.pyplot as plt
import torch

from generate import load_model, sample

# (label, run name) in the order they appear on the card
RUNS = [
    ("32 characters", "transformer_10000_ctx32"),
    ("64 characters", "transformer_10000"),
    ("128 characters", "transformer_10000_ctx128"),
    ("256 characters", "transformer_10000_ctx256"),
]
PROMPT = "Once upon a time"
TEMPERATURE = 0.5
MAX_NEW_CHARS = 220
SEED = 42
OUT = "results/context_card.png"

device = "cpu"
labels, losses, samples = [], [], []
for label, run in RUNS:
    try:
        with open(f"results/{run}.json", encoding="utf-8") as f:
            result = json.load(f)
        model, stoi, itos = load_model(run, device)
    except FileNotFoundError:
        print(f"skipping {run}: not trained yet")
        continue
    torch.manual_seed(SEED)
    labels.append(label)
    losses.append(result["final_val_loss"])
    text = sample(model, stoi, itos, PROMPT, TEMPERATURE, MAX_NEW_CHARS, device)
    end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind('." '))
    samples.append(text[: end + 1] if end > len(PROMPT) else text)

fig = plt.figure(figsize=(8, 10))
fig.suptitle("How far back can a tiny GPT look?", fontsize=18, fontweight="bold", y=0.975)
fig.text(0.5, 0.935, "Same model (~0.8M parameters), only the context window changes", ha="center", fontsize=11, color="#555")

chart = fig.add_axes([0.12, 0.66, 0.8, 0.24])
chart.plot(labels, losses, marker="o", color="#2b6cb0", linewidth=2)
for x, y in zip(labels, losses):
    chart.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
chart.set_ylabel("validation loss (lower is better)")
chart.set_xlabel("context window")
chart.grid(alpha=0.3)
chart.spines[["top", "right"]].set_visible(False)

top = 0.58
for label, text in zip(labels, samples):
    fig.text(0.06, top, label, fontsize=11, fontweight="bold", color="#2b6cb0")
    wrapped = textwrap.fill(f'"{text}"', width=95)
    fig.text(0.06, top - 0.012, wrapped, fontsize=9, va="top", family="DejaVu Serif")
    top -= 0.03 + 0.02 * wrapped.count("\n") + 0.02

fig.text(
    0.5, 0.025,
    f'Prompt "{PROMPT}", temperature {TEMPERATURE}, first sample from each model (not cherry-picked). '
    "Trained 10,000 steps on TinyStories on a MacBook Air M2.\n"
    "Longer contexts also see more text per step (batch of 32 windows).",
    ha="center", fontsize=8, color="#777", wrap=True,
)
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}")
