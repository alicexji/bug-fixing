from transformers import PreTrainedTokenizerFast

TOKENIZER_PATH = "tokenizer/hf_tokenizer"

def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

    print("Tokenizer loaded.")

    sample = "public int add(int a, int b) { return a + b; }"

    tokens = tokenizer.tokenize(sample)
    ids = tokenizer.encode(sample)

    print("\nSample input:")
    print(sample)

    print("\nTokens:")
    print(tokens[:20])

    print("\nToken IDs:")
    print(ids[:20])

    print("\nDecoded:")
    print(tokenizer.decode(ids))

    # special tokens
    print("\nSpecial tokens:")
    print("pad:", tokenizer.pad_token)
    print("eos:", tokenizer.eos_token)
    print("unk:", tokenizer.unk_token)

    # sentinel tokens
    print("\nSentinel tokens:")
    print("<extra_id_0>:", tokenizer.convert_tokens_to_ids("<extra_id_0>"))
    print("<extra_id_99>:", tokenizer.convert_tokens_to_ids("<extra_id_99>"))


if __name__ == "__main__":
    main()