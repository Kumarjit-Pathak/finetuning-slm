"""Dataset loading, formatting, chat templates, and quality checks."""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Optional


def _fallback_dataset(dataset_name: str) -> Dataset:
    """Return a small local fallback dataset when Hub access is unavailable."""
    if dataset_name == "databricks/dolly-15k":
        templates = [
            (
                "open_qa",
                "Explain what fine-tuning means for a language model in the context of {topic}.",
                "Fine-tuning adapts a pre-trained model to {topic} by continuing training on task-specific examples with a narrower objective.",
            ),
            (
                "closed_qa",
                "What role does LoRA play when adapting a model for {topic}?",
                "LoRA injects low-rank trainable matrices so the model can learn {topic} behavior with far fewer updated parameters.",
            ),
            (
                "summarization",
                "Summarize why data quality matters for {topic}.",
                "For {topic}, clean data improves the learning signal, reduces noise, and usually beats a much larger messy dataset.",
            ),
            (
                "classification",
                "Classify the sentiment of this feedback about {topic}: 'The workflow was clear and useful.'",
                "positive",
            ),
            (
                "brainstorming",
                "Suggest a lightweight evaluation metric for {topic}.",
                "Use a task-specific exact-match or validity metric for {topic}, then pair it with a small qualitative review set.",
            ),
        ]
        topics = [
            "instruction tuning",
            "domain adaptation",
            "tool calling",
            "safety evaluation",
            "clinical summarization",
            "code review",
            "tokenizer analysis",
            "preference optimization",
            "knowledge distillation",
            "parameter-efficient fine-tuning",
        ]
        rows = []
        for i in range(240):
            category, instruction_tmpl, response_tmpl = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            rows.append(
                {
                    "category": category,
                    "instruction": instruction_tmpl.format(topic=topic),
                    "response": response_tmpl.format(topic=topic),
                    "output": response_tmpl.format(topic=topic),
                }
            )
        return Dataset.from_list(rows)

    if dataset_name == "tatsu-lab/alpaca":
        templates = [
            (
                "Describe the difference between full fine-tuning and LoRA for {topic}.",
                "Full fine-tuning updates all weights for {topic}, while LoRA keeps the base model frozen and trains small low-rank adapters.",
            ),
            (
                "Give one reason to use synthetic data cautiously in {topic}.",
                "Synthetic data for {topic} can encode unrealistic patterns, so it still needs validation before training a production model.",
            ),
            (
                "Write a short instruction-following response about {topic}.",
                "{topic} benefits from explicit instructions, representative examples, and evaluation that matches the deployment scenario.",
            ),
            (
                "Explain prompt tuning in one sentence for {topic}.",
                "Prompt tuning learns a small set of virtual prompt embeddings for {topic} while keeping the rest of the model frozen.",
            ),
            (
                "Provide a concise tip for preventing overfitting in {topic}.",
                "Use a validation split, early stopping, and enough data diversity so the model does not memorize {topic} examples.",
            ),
        ]
        topics = [
            "medical summarization",
            "instruction tuning",
            "JSON tool calls",
            "sentiment classification",
            "adapter training",
            "small-model evaluation",
            "domain-specific QA",
            "safety alignment",
        ]
        rows = []
        for i in range(240):
            instruction_tmpl, output_tmpl = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            rows.append(
                {
                    "instruction": instruction_tmpl.format(topic=topic),
                    "input": "",
                    "output": output_tmpl.format(topic=topic),
                }
            )
        return Dataset.from_list(rows)

    raise RuntimeError(f"No local fallback is defined for dataset '{dataset_name}'.")


def load_and_format_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load a dataset from HuggingFace Hub with optional sampling."""
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as exc:
        print(
            f"Warning: failed to load '{dataset_name}' from the Hugging Face Hub "
            f"({type(exc).__name__}: {exc}). Using a local fallback sample instead."
        )
        ds = _fallback_dataset(dataset_name)
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
