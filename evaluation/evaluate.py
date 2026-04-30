import os
import json
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast

from codebleu import calc_codebleu

TEST_PATH = "data/processed/test.jsonl"

# CHANGE THIS per run
MODEL_PATH = "models/finetuned_pretrained"
#MODEL_PATH = "models/finetuned_scratch"

MAX_GEN_LEN = 256


def get_output_name(model_path):
    if "pretrained" in model_path:
        return "pretrained"
    elif "scratch" in model_path or "base" in model_path:
        return "scratch"
    else:
        return "unknown"


def load_data():
    data = []
    with open(TEST_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def exact_match(preds, refs):
    correct = 0
    for p, r in zip(preds, refs):
        if p.strip() == r.strip():
            correct += 1
    return correct / len(preds)


def generate_predictions(model, tokenizer, data, device):
    preds = []
    refs = []

    for item in tqdm(data):
        input_text = item["input"]
        target_text = item["output"]

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_GEN_LEN,
                num_beams=5,
                early_stopping=True
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        preds.append(pred)
        refs.append(target_text)

    return preds, refs


def compute_codebleu(preds, refs):
    refs_wrapped = [[r] for r in refs]

    try:
        result = calc_codebleu(
            refs_wrapped,
            preds,
            lang="java",
            weights=(0.5, 0.5, 0.0, 0.0) 
        )
        return result

    except Exception:
        print("⚠️ Falling back to simple BLEU (tree-sitter failed)")

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

        avg_score = sum(scores) / len(scores)

        return {
            "codebleu": avg_score,
            "ngram_match_score": avg_score,
            "weighted_ngram_match_score": avg_score,
            "syntax_match_score": 0.0,
            "dataflow_match_score": 0.0
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

    print("Loading test data...")
    data = load_data()

    print("Generating predictions...")
    preds, refs = generate_predictions(model, tokenizer, data, device)

    print("\nSample outputs:")
    for i in range(min(5, len(data))):
        print("INPUT:", data[i]["input"])
        print("PRED:", preds[i])
        print("REF:", refs[i])
        print("-----")

    print("Computing Exact Match...")
    em = exact_match(preds, refs)

    print("Computing CodeBLEU...")
    codebleu_result = compute_codebleu(preds, refs)

    print("\n===== RESULTS =====")
    print(f"Exact Match: {em:.4f}")
    print(f"CodeBLEU: {codebleu_result['codebleu']:.4f}")

    print("\nBreakdown:")
    for k, v in codebleu_result.items():
        if k != "codebleu":
            print(f"{k}: {v:.4f}")

    # SAVE OUTPUTS
    os.makedirs("outputs", exist_ok=True)

    tag = get_output_name(MODEL_PATH)

    json_file = f"outputs/preds_{tag}.json"
    txt_file = f"outputs/preds_{tag}.txt"

    # Save JSON
    with open(json_file, "w") as f:
        json.dump({
            "preds": preds,
            "refs": refs,
            "exact_match": em,
            "codebleu": codebleu_result
        }, f, indent=2)

    # Save readable text
    with open(txt_file, "w", encoding="utf-8") as f:
        for i in range(len(preds)):
            f.write(f"INPUT:\n{data[i]['input']}\n")
            f.write(f"PRED:\n{preds[i]}\n")
            f.write(f"REF:\n{refs[i]}\n")
            f.write("-----\n")

    print(f"\nSaved JSON results to {json_file}")
    print(f"Saved readable predictions to {txt_file}")


if __name__ == "__main__":
    main()