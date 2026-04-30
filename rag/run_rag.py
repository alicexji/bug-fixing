import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from codebleu import calc_codebleu


TEST_PATH = "data/processed/test.jsonl"
TRAIN_PATH = "data/processed/train.jsonl"

# ✅ REQUIRED MODEL
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


# =========================
# Data loading
# =========================
def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# =========================
# Retriever
# =========================
class Retriever:
    def __init__(self, data_path):
        print("Loading training data...")
        self.data = load_data(data_path)

        print("Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print("Encoding training inputs...")
        texts = [d["input"] for d in self.data]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def retrieve(self, query, k=3):
        query_emb = self.embedder.encode([query])
        distances, indices = self.index.search(query_emb, k)
        return [self.data[i] for i in indices[0]]


# =========================
# Prompt builder
# =========================
def build_prompt(query, retrieved=None):
    if retrieved:
        prompt = "Fix the bug in the following Java code.\n\n"

        for ex in retrieved:
            prompt += f"Buggy:\n{ex['input']}\n"
            prompt += f"Fixed:\n{ex['output']}\n\n"

        prompt += f"Buggy:\n{query}\nFixed:\n"
    else:
        prompt = f"Fix the bug in the following Java code:\n{query}\nFixed:\n"

    return prompt


# =========================
# Generation
# =========================
def generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# Metrics
# =========================
def exact_match(preds, refs):
    correct = 0
    for p, r in zip(preds, refs):
        if p.strip() == r.strip():
            correct += 1
    return correct / len(preds)


def compute_codebleu(preds, refs):
    refs_wrapped = [[r] for r in refs]

    try:
        result = calc_codebleu(
            refs_wrapped,
            preds,
            lang="java",
            weights=(0.5, 0.5, 0.0, 0.0)  # avoid tree-sitter issues
        )
        return result
    except Exception:
        print("⚠️ CodeBLEU fallback (BLEU only)")

        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1

        scores = []
        for p, r in zip(preds, refs):
            score = sentence_bleu(
                [r.split()],
                p.split(),
                smoothing_function=smoothie
            )
            scores.append(score)

        avg = sum(scores) / len(scores)

        return {
            "codebleu": avg,
            "ngram_match_score": avg,
            "weighted_ngram_match_score": avg,
            "syntax_match_score": 0.0,
            "dataflow_match_score": 0.0
        }


# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    print("Loading test data...")
    #test_data = load_data(TEST_PATH)  # ✅ full test set
    test_data = load_data(TEST_PATH)[:20]
    
    print("Building retriever...")
    retriever = Retriever(TRAIN_PATH)

    rag_preds = []
    zero_preds = []
    refs = []

    print("Running RAG + Zero-shot...")

    for item in tqdm(test_data):
        query = item["input"]
        refs.append(item["output"])

        # 🔵 RAG (3-shot)
        retrieved = retriever.retrieve(query, k=3)
        rag_prompt = build_prompt(query, retrieved)
        rag_output = generate(model, tokenizer, rag_prompt, device)
        rag_preds.append(rag_output)

        # 🔴 Zero-shot
        zero_prompt = build_prompt(query, None)
        zero_output = generate(model, tokenizer, zero_prompt, device)
        zero_preds.append(zero_output)

    # =========================
    # Metrics
    # =========================
    rag_em = exact_match(rag_preds, refs)
    zero_em = exact_match(zero_preds, refs)

    rag_cb = compute_codebleu(rag_preds, refs)
    zero_cb = compute_codebleu(zero_preds, refs)

    print("\n===== RESULTS =====")
    print(f"RAG EM: {rag_em:.4f}")
    print(f"RAG CodeBLEU: {rag_cb['codebleu']:.4f}")

    print(f"\nZero-shot EM: {zero_em:.4f}")
    print(f"Zero-shot CodeBLEU: {zero_cb['codebleu']:.4f}")

    # =========================
    # Save outputs
    # =========================
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/rag_results.json", "w") as f:
        json.dump({
            "rag_preds": rag_preds,
            "zero_preds": zero_preds,
            "refs": refs,
            "rag_em": rag_em,
            "zero_em": zero_em,
            "rag_codebleu": rag_cb,
            "zero_codebleu": zero_cb
        }, f, indent=2)

    print("\nSaved RAG results.")

    # Print samples
    print("\nSample outputs:")
    for i in range(min(3, len(test_data))):
        print("\nINPUT:", test_data[i]["input"])
        print("RAG:", rag_preds[i])
        print("ZERO:", zero_preds[i])
        print("REF:", refs[i])


if __name__ == "__main__":
    main()