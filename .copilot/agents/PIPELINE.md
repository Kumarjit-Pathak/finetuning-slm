# CLI Agent Pipeline — Orchestration Guide

## Quick Reference

This directory contains CLI-optimized prompt templates converted from `.github/agents/*.agent.md`.
Each template has placeholders (`{PHASE}`, `{STEP}`, `{NOTEBOOK_FILENAME}`, etc.) that get filled at invocation time.

---

## Pipeline Order (per step)

```
1. developer    → Implements notebook/code (general-purpose, sync)
2. reviewer     → Audits vs plan, PASS/FAIL report (general-purpose, sync)
3. [if FAIL]    → Developer fixes gaps, reviewer re-checks
4. documentation → Creates/updates HTML page (general-purpose, sync)
5. visualization → Creates SVG diagrams (general-purpose, background — parallel with humanizer prep)
6. humanizer    → Final tone polish on HTML (general-purpose, sync)
7. commit       → git add + commit with descriptive message
```

---

## Agent → Task Tool Mapping

| Agent | task `agent_type` | Recommended `model` | `mode` |
|-------|-------------------|---------------------|--------|
| developer | `general-purpose` | `claude-sonnet-4.5` | sync |
| reviewer | `general-purpose` | `claude-sonnet-4.5` | sync |
| documentation | `general-purpose` | `claude-sonnet-4.5` | sync |
| visualization | `general-purpose` | `claude-sonnet-4.5` | background (parallel OK) |
| humanizer | `general-purpose` | `claude-opus-4.6` | sync |

---

## Placeholders

| Placeholder | Example Value |
|-------------|---------------|
| `{PROJECT_ROOT}` | `/path/to/finetuning-slm` |
| `{PHASE}` | `1` |
| `{STEP}` | `3` |
| `{NOTEBOOK_FILENAME}` | `01_dataset_design_and_data_prep.ipynb` |
| `{HTML_FILENAME}` | `techniques.html` |
| `{TOPIC_NAME}` | `Dataset Design & Data Preparation` |
| `{SVG_FILENAMES}` | `dataset-design-flowchart.svg` |
| `{DIAGRAM_INSTRUCTIONS}` | Specific description of what to draw |

---

## Parallelization Opportunities

- **visualization** can run in background while **documentation** completes
- Multiple **visualization** SVGs can be launched in parallel
- **reviewer** must wait for **developer** to finish
- **humanizer** must wait for **documentation** to finish

---

## Error Handling

- If **reviewer** returns ⚠️ NEEDS REVISION → re-invoke **developer** with the gap list
- If **reviewer** returns ❌ REJECTED → major issues, re-read plan section carefully
- Max 2 revision cycles before escalating to user for guidance
