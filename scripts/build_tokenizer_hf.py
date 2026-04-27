import os
import sentencepiece as spm
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

SPM_PATH = "tokenizer/spm.model"
OUTPUT_DIR = "tokenizer/hf_tokenizer"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading SentencePiece model...")

    # Load SentencePiece into HF-compatible tokenizer
    tokenizer = SentencePieceUnigramTokenizer.from_spm(SPM_PATH)

    print("Wrapping into PreTrainedTokenizerFast...")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="<s>"
    )

    # Add sentinel tokens manually
    extra_tokens = [f"<extra_id_{i}>" for i in range(100)]
    hf_tokenizer.add_special_tokens({
        "additional_special_tokens": extra_tokens
    })

    hf_tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSaved tokenizer to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()