import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm

from pretraining.dataset import SpanCorruptionDataset


DATA_PATH = "data/raw/pretrain_methods.txt"
MODEL_PATH = "models/base_model"
OUTPUT_DIR = "models/pretrained_model"

BATCH_SIZE = 16
EPOCHS = 3        
MAX_LENGTH = 256
LR = 5e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        texts = f.readlines()

    texts = [t.strip() for t in texts if t.strip()]

    # ~50K requirement
    texts = texts[:50000]

    print(f"Loaded {len(texts)} samples")
    return texts


def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_ids": input_ids,
        "labels": labels
    }


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading tokenizer + model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

    print("Loading data...")
    texts = load_data()

    dataset = SpanCorruptionDataset(
        texts,
        tokenizer,
        max_length=MAX_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,       
        pin_memory=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            loop.set_postfix({
                "loss": f"{loss.item():.3f}",
                "avg_loss": f"{(total_loss / (step + 1)):.3f}"
            })

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nPretraining complete. Model saved.")


if __name__ == "__main__":
    main()