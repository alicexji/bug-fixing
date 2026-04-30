import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from tqdm import tqdm

from finetuning.dataset import BugFixDataset


DATA_PATH = "data/processed/train.jsonl"

# CHANGE THIS for Pipeline A vs B
#MODEL_PATH = "models/pretrained_model"   # Pipeline A
MODEL_PATH = "models/base_model"      # Pipeline B

# CHANGE THIS for Pipeline A vs B
#OUTPUT_DIR = "models/finetuned_pretrained"
OUTPUT_DIR = "models/finetuned_scratch"

BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4


def load_data():
    data = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer + model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

    print("Loading data...")
    data = load_data()
    dataset = BugFixDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Starting fine-tuning...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()