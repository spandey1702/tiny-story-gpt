import argparse

# The first 5,000 stories always give the same split: 4,500 to train on and
# 500 held out for validation. Extra stories only ever come from after story
# 5,000, so they can never overlap the validation set.
BASE_STORIES = 5000
TRAIN_STORIES = 4500

# The raw TinyStories download uses typographic quotes, dashes and ellipses;
# the original 5k file has plain ASCII versions. Mapping them keeps the
# vocabulary the same (70 characters) and reproduces the original files exactly
CLEAN = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                       "—": "-", "–": "-", "…": "..."})

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data/tinystories_5k.txt")
parser.add_argument("--extra-stories", type=int, default=0,
                    help="stories after the first 5,000 to add to a larger training file")
args = parser.parse_args()

with open(args.source, encoding="utf-8") as f:
    stories = [s.strip() for s in f.read().translate(CLEAN).split("<|endoftext|>") if s.strip()]

needed = BASE_STORIES + args.extra_stories
# A download cut at a byte limit ends mid-story, so the last story is only
# trusted if the file holds more stories than we use
if len(stories) <= needed and args.extra_stories:
    raise SystemExit(f"{args.source} has {len(stories)} stories; need more than {needed}")

train, valid = stories[:TRAIN_STORIES], stories[TRAIN_STORIES:BASE_STORIES]
outputs = [("data/train.txt", train), ("data/valid.txt", valid)]
if args.extra_stories:
    big = train + stories[BASE_STORIES:needed]
    outputs.append((f"data/train_{len(big)}.txt", big))

for path, chunk in outputs:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunk))
    print(f"Wrote {len(chunk):,} stories to {path}")
