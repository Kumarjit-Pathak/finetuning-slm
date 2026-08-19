---
description: "SLM fine-tuning Developer agent. Use when: implementing notebook content, writing Python code for fine-tuning techniques, creating training pipelines, building utility functions, implementing use cases. Develops code based on the slm-finetuning-plan.md requirements."
tools: [read, edit, search, execute, agent]
---

Implement notebooks and Python code per `slm-finetuning-plan.md`.

## Rules
- Read the relevant plan section before coding
- Cell order: setup → concept → implementation → failure points → comparison
- Include Colab-compatible setup cell at the top of every notebook
- Provide `low_resource` config alternatives for 6GB GPU users
- Use SmolLM2 / Qwen2.5 model families; no hardcoded paths
- Use `src/training_utils.py`, `src/visualization.py` — don't duplicate
- Never skip failure point demonstrations
