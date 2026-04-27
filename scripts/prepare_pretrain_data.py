import os
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_PATH = "data/raw/pretrain_methods.txt"
NUM_SAMPLES = 50000


def main():
    print("Loading CodeSearchNet Java dataset...")
    csn = load_dataset("code_search_net", "java")

    print(f"Sampling {NUM_SAMPLES} methods...")
    methods = csn["train"].shuffle(seed=42).select(range(NUM_SAMPLES))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Writing methods to file...")
    count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for m in tqdm(methods):
            code = m.get("whole_func_string", "")

            if not code.strip():
                continue

            # normalize whitespace (important for tokenizer training later)
            code = code.replace("\n", " ").strip()

            f.write(code + "\n")
            count += 1

    print("\nDone.")
    print(f"Saved {count} methods to {OUTPUT_PATH}")

    # sanity check
    print("\nSample output:")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for _ in range(3):
            print(f.readline().strip())


if __name__ == "__main__":
    main()