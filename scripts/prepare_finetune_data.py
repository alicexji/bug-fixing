import os
import json
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "data/processed"


def save_split(split, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    count = 0

    with open(path, "w", encoding="utf-8") as f:
        for ex in tqdm(split):
            buggy = ex.get("buggy", "").strip()
            fixed = ex.get("fixed", "").strip()

            if not buggy or not fixed:
                continue

            json.dump({
                "input": buggy,
                "output": fixed
            }, f)
            f.write("\n")
            count += 1

    print(f"Saved {count} examples to {path}")


def main():
    print("Loading CodeXGLUE bug-fix dataset...")
    dataset = load_dataset(
        "google/code_x_glue_cc_code_refinement",
        name="medium"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nProcessing splits...")
    save_split(dataset["train"], "train.jsonl")
    save_split(dataset["validation"], "val.jsonl")
    save_split(dataset["test"], "test.jsonl")

    print("\nDone.")

    # sanity check
    sample_file = os.path.join(OUTPUT_DIR, "train.jsonl")
    print("\nSample output:")
    with open(sample_file, "r", encoding="utf-8") as f:
        for _ in range(2):
            print(f.readline().strip())


if __name__ == "__main__":
    main()