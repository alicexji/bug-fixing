import os
from transformers import T5Config, T5ForConditionalGeneration, PreTrainedTokenizerFast

TOKENIZER_PATH = "tokenizer/hf_tokenizer"
OUTPUT_DIR = "models/base_model"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

    print("Tokenizer vocab size:", len(tokenizer))

    print("\nCreating T5 config...")

    config = T5Config(
        vocab_size=len(tokenizer),

        d_model=512,
        d_ff=2048,
        d_kv=64,

        num_heads=8,
        num_layers=6,
        num_decoder_layers=6,

        # match tokenizer
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id
    )

    print("\nInitializing model from scratch...")
    model = T5ForConditionalGeneration(config)

    print("Resizing embeddings...")
    model.resize_token_embeddings(len(tokenizer))

    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone.")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()