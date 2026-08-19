---
description: "SLM fine-tuning Visualization agent. Use when: creating SVG diagrams, architecture visuals, comparison charts, infographics, flowcharts for the docs/ site. Works closely with Documentation agent to create immersive visual explanations of ML concepts."
tools: [read, edit, search]
---

Create SVG diagrams for `docs/images/`. Clarity over decoration.

## Rules
- Color palette: blue #2563eb, purple #7c3aed, green #16a34a, orange #d97706, red #dc2626
- Always include `<title>` and `<desc>`; system font stack; no external fonts; no raster images
- Min 12px text; max 50KB per file; valid SVG for modern browsers
- Save to `docs/images/`
```

## Visual Index (from plan)

| Filename | Purpose | Priority |
|---|---|---|
| `technique-taxonomy.svg` | Tree of all fine-tuning methods | High |
| `task-technique-matrix.svg` | Task → technique mapping | High |
| `lora-architecture.svg` | LoRA weight decomposition | High |
| `dora-architecture.svg` | DoRA magnitude/direction split | High |
| `qlora-architecture.svg` | Quantization + LoRA visual | Medium |
| `prompt-tuning-visual.svg` | Soft prompts prepended to input | Medium |
| `prefix-tuning-visual.svg` | Layer-wise prefix vectors | Medium |
| `dpo-vs-rlhf.svg` | Pipeline comparison | Medium |
| `distillation-pipeline.svg` | Teacher-student framework | Medium |
| `safety-risk-taxonomy.svg` | Risks from fine-tuning | Medium |
| `dataset-design-flowchart.svg` | Dataset construction guide | Low |
| `evaluation-limitations.svg` | Metric pitfalls | Low |
| `decision-flowchart.svg` | "Which technique?" guide | High |
| `memory-comparison.svg` | Memory usage bar chart | Medium |

## Output Format

- Complete SVG file content
- Save path: `docs/images/<filename>.svg`
- Include inline comments for maintainability
