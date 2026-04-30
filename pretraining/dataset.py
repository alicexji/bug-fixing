import random
from torch.utils.data import Dataset


class SpanCorruptionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # ✅ Keep tokenization here (better for CPU/Windows)
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_length
        )

        input_ids, labels = self.apply_span_corruption(tokens)

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def apply_span_corruption(self, tokens):
        input_ids = []
        labels = []

        i = 0
        sentinel_id = 0

        while i < len(tokens):
            if random.random() < self.mask_prob:
                span_length = random.randint(1, 5)
                span = tokens[i:i + span_length]

                sentinel_token = self.tokenizer.convert_tokens_to_ids(
                    f"<extra_id_{sentinel_id}>"
                )

                input_ids.append(sentinel_token)

                labels.append(sentinel_token)
                labels.extend(span)

                sentinel_id += 1
                i += span_length
            else:
                input_ids.append(tokens[i])
                i += 1

        # ✅ IMPORTANT for T5 training
        labels.append(self.tokenizer.eos_token_id)

        return input_ids, labels