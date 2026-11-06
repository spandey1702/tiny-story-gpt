import json
import textwrap

import matplotlib.pyplot as plt
import torch

from device import get_device
from generate import load_model, sample

# (label, results file) top to bottom; losses come from the full-validation scoreboard
RUNS = [
    ("Baseline\n0.84M params, 4,500 stories", None, 0.8410),
    ("A: + cosine LR schedule", "runA_cosine", None),
    ("B: 11x more stories", "runB_data50k", None),
    ("C: + 3x bigger model\n(cosine)", "runC_bigger", None),
    ("C: + 3x bigger model\n(constant LR)", "runC_bigger_constant", None),
]
SAMPLES = [("Before: baseline (0.841)", "transformer_10000_ctx256"), ("After: Run C (0.703)", "runC_bigger")]
PROMPT = "Once upon a time"
TEMPERATURE = 0.8
MAX_NEW_CHARS = 400
SEED = 42
OUT = "results/runs_card.png"
BLUE, RED, GREY = "#2b6cb0", "#c53030", "#555"

labels, losses = [], []
for label, run, loss in RUNS:
    if run:
        with open(f"results/{run}.json", encoding="utf-8") as f:
            loss = json.load(f)["full_val_loss"]
    labels.append(label)
    losses.append(loss)

device = get_device()
stories = []
for label, run in SAMPLES:
    torch.manual_seed(SEED)
    model, stoi, itos = load_model(run, device)
    text = sample(model, stoi, itos, PROMPT, TEMPERATURE, MAX_NEW_CHARS, device).replace("\n", " ")
    end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind('." '))
    stories.append((label, text[: end + 1] if end > len(PROMPT) else text))

fig = plt.figure(figsize=(8, 10))
fig.suptitle("What actually improved my tiny story model?", fontsize=18, fontweight="bold", y=0.975)
fig.text(0.5, 0.935, "One change at a time, scored on the same 500 held-out stories",
         ha="center", fontsize=11, color=GREY)

chart = fig.add_axes([0.36, 0.6, 0.56, 0.3])
y = list(range(len(labels)))[::-1]
baseline = losses[0]
colors = [GREY] + [RED if l > baseline else BLUE for l in losses[1:]]
chart.scatter(losses, y, color=colors, s=90, zorder=3)
chart.axvline(baseline, color=GREY, linestyle=":", linewidth=1)
for yi, loss in zip(y, losses):
    change = "" if loss == baseline else f"  ({(loss - baseline) / baseline * 100:+.1f}%)"
    chart.annotate(f"{loss:.4f}{change}", (loss, yi), textcoords="offset points", xytext=(8, -4), fontsize=10)
chart.set_yticks(y, labels, fontsize=10)
chart.set_xlim(0.65, 0.95)
chart.set_xlabel("validation loss (lower is better)")
chart.grid(alpha=0.3)
chart.spines[["top", "right"]].set_visible(False)

top = 0.49
fig.text(0.06, top + 0.025, f'Same prompt ("{PROMPT}") and seed, first sample from each model',
         fontsize=11, fontweight="bold")
for (label, text), color in zip(stories, [GREY, BLUE]):
    fig.text(0.06, top, label, fontsize=11, fontweight="bold", color=color)
    wrapped = textwrap.fill(f'"{text}"', width=82)
    fig.text(0.06, top - 0.016, wrapped, fontsize=10.5, va="top", family="DejaVu Serif")
    top -= 0.04 + 0.024 * wrapped.count("\n") + 0.04

fig.text(
    0.5, 0.025,
    "Character-level transformers trained 10,000 steps on TinyStories (baseline on a MacBook Air M2,\n"
    f"runs A to C on a free Colab T4). Temperature {TEMPERATURE}, not cherry-picked. "
    "The After sample is from the cosine version of Run C.",
    ha="center", fontsize=8, color="#777",
)
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}")
