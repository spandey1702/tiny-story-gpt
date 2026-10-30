SOURCE = "data/tinystories_5k.txt"

with open(SOURCE, encoding="utf-8") as f:
    stories = [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()]

split = int(len(stories) * 0.9)
for path, chunk in [("data/train.txt", stories[:split]), ("data/valid.txt", stories[split:])]:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunk))
    print(f"Wrote {len(chunk)} stories to {path}")
