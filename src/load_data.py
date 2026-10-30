import torch


def load_data_from_files(train_path, val_path):
    with open(train_path, encoding="utf-8") as f:
        train_text = f.read()
    with open(val_path, encoding="utf-8") as f:
        val_text = f.read()

    chars = sorted(set(train_text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    train_data = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    val_data = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)

    return train_data, val_data, stoi, itos, vocab_size


def get_batch(data, block_size, batch_size):
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y
