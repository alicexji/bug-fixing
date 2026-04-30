import os
import sentencepiece as spm

INPUT_FILE = "data/raw/pretrain_methods.txt"
MODEL_PREFIX = "tokenizer/spm"
VOCAB_SIZE = 16384


def main():
    os.makedirs("tokenizer", exist_ok=True)

    print("Training SentencePiece tokenizer...")

    spm.SentencePieceTrainer.train(
        input=INPUT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",

        character_coverage=1.0,

        # CLEAN SPECIAL TOKENS
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",

        #for T5 span corruption
        user_defined_symbols=[f"<extra_id_{i}>" for i in range(100)],

   
        normalization_rule_name="identity"
    )

    print("\nTokenizer training complete.")
    print("Saved to tokenizer/")


if __name__ == "__main__":
    main()