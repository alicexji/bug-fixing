import torch
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast

MODEL_PATH = "models/base_model"


def main():
    print("Loading model + tokenizer...")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    model.eval()

    sample = "fix this bug: public int add(int a, int b) { return a - b; }"

    inputs = tokenizer(sample, return_tensors="pt")

    print("\nRunning forward pass...")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("\nInput:")
    print(sample)

    print("\nOutput:")
    print(decoded)
    


if __name__ == "__main__":
    main()