import sys

import torch

from model import BigramModel, TransformerModel


def load_model(run_name, device):
    checkpoint = torch.load(f"checkpoints/{run_name}.pt", map_location=device)
    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    if checkpoint["model"] == "bigram":
        model = BigramModel(len(stoi))
    else:
        model = TransformerModel(len(stoi), checkpoint["block_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, stoi, itos


def sample(model, stoi, itos, prompt, temperature, max_new_chars, device):
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)
    out = model.generate(idx, max_new_chars, temperature=temperature)[0].tolist()
    return "".join(itos[i] for i in out).split("\n\n")[0]


if __name__ == "__main__":
    RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "transformer_10000"
    TEMPERATURE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    SAMPLES_PER_PROMPT = 5
    MAX_NEW_CHARS = 250
    PROMPTS = ["Once upon a time", "One day, Lily", "Tom was sad because"]

    torch.manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, stoi, itos = load_model(RUN_NAME, device)

    print(f"{RUN_NAME} | temperature {TEMPERATURE}")
    for prompt in PROMPTS:
        print(f"\n=== {prompt} ===")
        for n in range(1, SAMPLES_PER_PROMPT + 1):
            print(f"[{n}] {sample(model, stoi, itos, prompt, TEMPERATURE, MAX_NEW_CHARS, device)}\n")
