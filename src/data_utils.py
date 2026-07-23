"""Dataset loading, formatting, chat templates, and quality checks."""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Optional


def load_and_format_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load a dataset from HuggingFace Hub with optional sampling."""
    ds = load_dataset(dataset_name, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


def apply_chat_template(
    examples: dict,
    tokenizer: AutoTokenizer,
    system_prompt: str = "",
) -> dict:
    """Apply chat template formatting to a batch of examples."""
    formatted = []
    for instruction, response in zip(examples["instruction"], examples["output"]):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": response})
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append(text)
    return {"text": formatted}


def create_preference_pairs(
    examples: dict,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> dict:
    """Format data for DPO training with chosen/rejected pairs."""
    return {
        "prompt": examples["prompt"],
        "chosen": examples[chosen_key],
        "rejected": examples[rejected_key],
    }


def compute_dataset_statistics(dataset: Dataset) -> dict:
    """Compute basic statistics about a dataset."""
    lengths = [len(str(ex)) for ex in dataset["text"] if "text" in dataset.column_names]
    if not lengths:
        return {"num_examples": len(dataset), "columns": dataset.column_names}
    return {
        "num_examples": len(dataset),
        "columns": dataset.column_names,
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def check_data_quality(dataset: Dataset, text_column: str = "text") -> dict:
    """Run basic quality checks on a dataset."""
    issues = []
    texts = dataset[text_column] if text_column in dataset.column_names else []

    # Check for duplicates
    unique = set(texts)
    if len(unique) < len(texts):
        issues.append(f"Found {len(texts) - len(unique)} duplicate entries")

    # Check for empty entries
    empty_count = sum(1 for t in texts if not t or not t.strip())
    if empty_count:
        issues.append(f"Found {empty_count} empty entries")

    return {"total": len(texts), "unique": len(unique), "empty": empty_count, "issues": issues}
