import glob
import json

import matplotlib.pyplot as plt

runs = []
for path in sorted(glob.glob("results/*.json")):
    with open(path, encoding="utf-8") as f:
        runs.append(json.load(f))

print(f"{'model':<14}{'parameters':>12}{'val loss':>10}{'seconds':>10}")
for run in runs:
    print(f"{run['run']:<14}{run['parameters']:>12,}{run['final_val_loss']:>10.3f}{run['train_seconds']:>10.1f}")

for run in runs:
    print(f"\n--- {run['run']} sample ---")
    print(run["sample"][:300])

plt.figure(figsize=(8, 5))
for run in runs:
    steps = [h["step"] for h in run["history"]]
    val = [h["val"] for h in run["history"]]
    plt.plot(steps, val, label=f"{run['run']} (val {run['final_val_loss']:.2f})")
plt.xlabel("training step")
plt.ylabel("validation loss (lower is better)")
plt.title("Tiny Story GPT: validation loss on TinyStories")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/loss_comparison.png", dpi=200)
print("\nSaved results/loss_comparison.png")
