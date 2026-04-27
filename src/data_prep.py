import random
from datasets import load_dataset

PREFIXES = [
    "Level: Age3-4 — Simple words. ",
    "Level: Age5-6 — Short sentences. ",
    "Level: Age7-8 — Moderate vocabulary. ",
    "Level: Age9-10 — Longer sentences. ",
    "Level: Age11-12 — Richer vocabulary. "
]


def sample_and_prefix(output_path: str, hf_name: str, sample_size: int = 50000, seed: int = 42):
    ds = load_dataset(hf_name, split="train")
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(sample_size, len(ds))))

    def add_prefix(example, idx):
        prefix = random.choice(PREFIXES)
        text = example.get("text", "")
        return {"text": prefix + text}

    ds = ds.map(add_prefix, with_indices=True)
    ds.to_json(output_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/processed/prefix_stories.json")
    p.add_argument("--hf", default="ajibawa-2023/Children-Stories-Collection")
    p.add_argument("--n", type=int, default=50000)
    args = p.parse_args()
    sample_and_prefix(args.output, args.hf, args.n)
