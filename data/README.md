# Data Directory

This directory is for locally cached datasets. Most datasets are loaded directly from HuggingFace Hub during notebook execution.

## Datasets Used

| Dataset | Source | Notebooks | License |
|---|---|---|---|
| databricks/dolly-15k | HF Hub | 02, 04, 05 | CC-BY-SA-3.0 |
| tatsu-lab/alpaca | HF Hub | 01, 04 | CC-BY-NC-4.0 |
| stanfordnlp/sst2 | HF Hub | 03 | MIT |
| Custom synthetic | Generated in notebooks | 14–18 | Created by us |

## Data Governance Notes

- **No real patient data** is used anywhere in this tutorial
- Clinical notes in notebook 18 are **synthetically generated** from public medical ontologies
- Code review data in notebook 15 is **synthetically generated**
- All HuggingFace datasets are loaded with their original licenses respected
- PII detection should be run on any custom datasets before fine-tuning (see notebook 01, Part C)

## Downloading Datasets

Datasets are automatically cached to `~/.cache/huggingface/datasets/` on first load.
No manual download is required.
