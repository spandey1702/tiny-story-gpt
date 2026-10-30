# tiny-story-gpt

A small character-level GPT trained from scratch on children's stories, on a MacBook Air (M2, 8 GB).
It compares a bigram baseline with a 4-layer transformer trained on the same data.

## Results

Trained on 4,500 TinyStories, validated on 500 held-out stories (character-level, 70-character vocabulary).

| model | parameters | steps | val loss | training time (M2 GPU) |
|---|---|---|---|---|
| bigram | 4,900 | 3,000 | 2.300 | 6 s |
| transformer | 817,990 | 3,000 | 1.099 | 74 s |
| transformer | 817,990 | 10,000 | 0.978 | 247 s |

![Validation loss](results/loss_comparison.png)

Sample from the bigram model:

```
On. ace ssit way, ar Thay tl rnanthayud ake plliean.
```

Sample from the 10,000-step transformer:

```
She asked her room tree and had a big, yest.
"LeNo, I'm sharprise you did and make this."
```

The transformer learned spelling, names, dialogue and sentence rhythm, but not meaning yet.
At 0.8M parameters with a 64-character context, it is too small to keep a story coherent.

GPU usage during the 10,000-step run is logged in `results/gpu_usage_10000.csv`
(about 92% utilization while training, about 120 MB allocated by PyTorch).

## Model

- Bigram: a 70 x 70 lookup table predicting the next character from the current one.
- Transformer: 4 blocks of 4-head causal self-attention and a feed-forward layer, 128-dim embeddings,
  64-character context, dropout 0.1, AdamW at 1e-3, batch size 32.

## Project layout

```
src/prepare_data.py   split the stories 90/10 into data/train.txt and data/valid.txt
src/load_data.py      character vocabulary and random batches
src/model.py          BigramModel and TransformerModel
src/train.py          training loop; set MODEL and STEPS at the top
src/compare.py        comparison table and results/loss_comparison.png
results/              per-run results (JSON), loss chart, GPU log
```

## Run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the first 5,000 stories from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
in `data/tinystories_5k.txt`, separated by `<|endoftext|>`, then:

```bash
python src/prepare_data.py
python src/train.py
python src/compare.py
```

## Credits

- Data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li, licensed under CDLA-Sharing-1.0. The data is not included in this repo.
- Model structure follows Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).
