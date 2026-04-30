from torch.utils.data import Dataset


class BugFixDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_text = item["input"]
        target_text = item["output"]

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        targets = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding=False
        )

        return {
            "input_ids": inputs["input_ids"],
            "labels": targets["input_ids"]
        }