"""Evaluation metrics: ROUGE, BERTScore, perplexity, JSON validity, OOD testing."""

import json
import math
import torch
from typing import Optional


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    import evaluate
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results


def compute_bertscore(predictions: list[str], references: list[str], model_type: str = "microsoft/deberta-xlarge-mnli") -> dict:
    """Compute BERTScore (precision, recall, F1)."""
    from bert_score import score
    P, R, F1 = score(predictions, references, model_type=model_type, verbose=False)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """Compute perplexity of a model on a list of texts."""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def evaluate_json_validity(outputs: list[str]) -> dict:
    """Evaluate JSON validity rate for tool-calling outputs."""
    valid = 0
    parse_errors = []

    for i, output in enumerate(outputs):
        try:
            json.loads(output)
            valid += 1
        except json.JSONDecodeError as e:
            parse_errors.append({"index": i, "error": str(e)[:100]})

    return {
        "total": len(outputs),
        "valid": valid,
        "validity_rate": valid / len(outputs) if outputs else 0,
        "sample_errors": parse_errors[:5],
    }


def evaluate_tool_call_accuracy(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Evaluate tool-calling accuracy (tool name, parameters)."""
    correct_tool = 0
    correct_params = 0

    for pred, gt in zip(predictions, ground_truth):
        if pred.get("name") == gt.get("name"):
            correct_tool += 1
            if pred.get("arguments") == gt.get("arguments"):
                correct_params += 1

    n = len(predictions)
    return {
        "tool_accuracy": correct_tool / n if n else 0,
        "param_accuracy": correct_params / n if n else 0,
        "total": n,
    }


def compute_ood_degradation(
    model,
    tokenizer,
    id_texts: list[str],
    ood_texts: list[str],
    max_length: int = 512,
) -> dict:
    """Compare in-domain vs out-of-domain perplexity to measure specialization tax."""
    id_ppl = compute_perplexity(model, tokenizer, id_texts, max_length)
    ood_ppl = compute_perplexity(model, tokenizer, ood_texts, max_length)
    return {
        "in_domain_perplexity": id_ppl,
        "out_of_domain_perplexity": ood_ppl,
        "degradation_ratio": ood_ppl / id_ppl if id_ppl > 0 else float("inf"),
    }
