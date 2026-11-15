# Project Understanding: finetuning-slm

> Comprehensive analysis of the repository, its architecture, current state, and remaining work.

---

## 1. What This Project Is

**finetuning-slm** is an end-to-end educational tutorial on fine-tuning Small Language Models (SLMs). It is designed as an academia-ready, hands-on resource covering 10+ fine-tuning techniques across 20 Jupyter notebooks, supported by a Python utility library, YAML configs, and a full interactive GitHub Pages documentation site.

**Author**: Kumarjit Pathak  
**License**: MIT  
**Primary development machine**: Apple M4 Max (36+ GB unified memory)  
**Target audience**: ML practitioners, students, and researchers wanting to learn SLM fine-tuning from basics to advanced techniques.

---

## 2. Supported Hardware & Environments

The project is designed to run across a wide spectrum of hardware:

| Tier | Examples | Capability |
|---|---|---|
| 6 GB GPU | RTX 3060, GTX 1660, Colab T4 (free) | SmolLM2-135M (all techniques), 360M/1.5B (QLoRA only) |
| 16 GB | Apple M1/M2, RTX 4060 Ti | SmolLM2-360M (LoRA/DoRA), Qwen2.5-1.5B (QLoRA) |
| 36 GB | Apple M3/M4, RTX 3090 | All notebooks including Qwen2.5-1.5B LoRA |
| 64+ GB | Apple M-series (64GB+), A100 | Everything at full precision |

**Design principle**: SmolLM2-135M runs ALL techniques on any hardware, so no learner is locked out. Every notebook has a low-resource config fallback.

---

## 3. Models Used

### Open Models (no authentication needed)
- **SmolLM2-135M / 135M-Instruct** -- Quick demos, fits any hardware
- **SmolLM2-360M-Instruct** -- Main tutorial model
- **Qwen2.5-1.5B-Instruct** -- Quality-critical demos
- **Qwen2.5-3B-Instruct** -- Distillation teacher

### Gated Models (HuggingFace token required, optional)
- Llama 3.2 1B / 3B
- Gemma 2 2B

The tutorial works fully with open models only. Gated models are optional drop-in alternatives.

---

## 4. Project Structure

```
finetuning-slm/
├── README.md                   # Project overview, quickstart, hardware matrix
├── slm-finetuning-plan.md      # Master plan (750+ lines) -- phases, progress tracker, key decisions
├── AGENTS.md                   # Agent pipeline summary
├── pyproject.toml              # Python deps (torch, transformers, peft, trl, etc.)
├── setup.sh                    # One-command env setup (uv + venv + kernel + verify)
├── .gitignore                  # Ignores outputs/, .venv/, model weights, checkpoints
│
├── notebooks/                  # 20 Jupyter notebooks (00-19) + result PNGs (18 images)
├── src/                        # Reusable Python utility library (7 files)
├── configs/                    # YAML training configs (7 files)
├── data/                       # Dataset documentation (README.md only, data loaded from HF Hub)
├── scripts/                    # download_models.py (pre-cache models for offline use)
├── docs/                       # GitHub Pages site (10 HTML pages + CSS + JS + images)
│   ├── css/                    # style.css (14KB), print.css (5KB)
│   ├── js/                     # main.js, search.js, search-index.js
│   ├── images/
│   │   ├── diagrams/           # 7 custom SVGs (LoRA arch, DoRA, QLoRA flow, DPO, etc.)
│   │   └── notebooks/          # 18 result PNGs (copied from notebooks/)
│   └── *.html                  # 10 pages: index, guide, techniques, comparison, etc.
│
├── .github/
│   ├── agents/                 # 5 agent definitions (developer, reviewer, documentation, visualization, humanizer)
│   └── workflows/
│       └── pages.yml           # GitHub Pages deploy with sanity checks (emoji/URL audit, search index rebuild)
│
├── .copilot/agents/            # (empty)
└── .tmp/executed/              # (empty, likely temp execution artifacts)
```

---

## 5. The 20 Notebooks

Each notebook follows a consistent structure: setup cell (Colab-compatible) -> concept explanation -> implementation -> failure points -> comparison/visualization.

| # | Notebook | Technique / Topic | Phase |
|---|---|---|---|
| 00 | `environment_setup` | Verify env, MPS/CUDA detection, CPU vs GPU benchmark | 1 |
| 01 | `dataset_design_and_data_prep` | Dataset design principles, tokenizers, chat templates, quality checks | 1 |
| 02 | `full_finetuning` | Full fine-tuning of SmolLM2-135M on dolly-15k | 2 |
| 03 | `feature_extraction_linear_probing` | Frozen base + classification head (SST-2 sentiment) | 2 |
| 04 | `lora_and_dora_finetuning` | LoRA + DoRA comparison on SmolLM2-360M | 3 |
| 05 | `qlora_finetuning` | 4-bit QLoRA on Qwen2.5-1.5B | 3 |
| 06 | `prompt_tuning` | Learnable soft prompts | 4 |
| 07 | `prefix_tuning` | Layer-wise prefix tokens | 4 |
| 08 | `dpo_alignment` | Direct Preference Optimization | 5 |
| 09 | `knowledge_distillation` | Teacher-student distillation | 5 |
| 10 | `model_merging` | SLERP/TIES/DARE merging | 5 |
| 11 | `failure_points_and_debugging` | Common pitfalls catalog | 7 |
| 12 | `responsible_safe_finetuning` | Safety guardrails, red-teaming | 6 |
| 13 | `evaluation_theory_and_benchmarking` | ROUGE, BERTScore, perplexity, ID vs OOD | 8 |
| 14 | `fun_shakespearean_insult_generator` | Fun use case: Shakespeare-style insults | 9 |
| 15 | `fun_code_roast_reviewer` | Fun use case: code roast/review | 9 |
| 16 | `fun_dungeon_master` | Fun use case: D&D dungeon master | 9 |
| 17 | `industry_agentic_tool_calling` | Industry use case: JSON tool calling | 10 |
| 18 | `industry_privacy_preserving_clinical_summarizer` | Industry use case: clinical note summarization | 10 |
| 19 | `apple_silicon_optimization_guide` | MPS vs MLX benchmark, Apple Silicon tips | 11 |

All 20 notebooks have been executed and verified end-to-end on M4 Max.

---

## 6. Source Library (`src/`)

The `src/` package provides reusable utilities consumed by the notebooks:

### `data_utils.py`
- `load_and_format_dataset()` -- loads from HF Hub with local fallback (240-row synthetic datasets for dolly-15k and alpaca)
- `apply_chat_template()` -- ChatML formatting with system prompt support
- `create_preference_pairs()` -- formats data for DPO (chosen/rejected pairs)
- `compute_dataset_statistics()` -- basic stats (lengths, counts)
- `check_data_quality()` -- dedup and empty-entry checks

### `training_utils.py`
- `setup_device()` -- MPS > CUDA > CPU auto-detection
- `get_device_info()` -- detailed backend info (VRAM, unified memory)
- `get_training_args()` -- sensible TrainingArguments defaults with auto precision (bf16 on MPS, fp16 on CUDA) and low-resource overrides
- `MemoryMonitorCallback` -- logs GPU memory every 50 steps
- `TrainingMetricsCallback` -- captures loss/step/timestamp for visualization

### `evaluation_utils.py`
- `compute_rouge()`, `compute_bertscore()`, `compute_perplexity()`
- `evaluate_json_validity()` -- JSON parse rate for tool-calling outputs
- `evaluate_tool_call_accuracy()` -- tool name + parameter exact match
- `compute_ood_degradation()` -- in-domain vs out-of-domain perplexity ratio

### `safety_utils.py`
- `RED_TEAM_CATEGORIES` -- 4 categories (direct harm, social engineering, bias, misinformation) with 3 prompts each
- `red_team_prompts()` -- retrieve prompts by category
- `test_guardrail_retention()` -- measure refusal rate via keyword detection
- `measure_refusal_rate()` -- standalone refusal rate from response list

### `visualization.py`
- `plot_training_curves()` -- loss over steps
- `plot_technique_comparison_radar()` -- multi-technique radar chart
- `plot_memory_comparison()` -- bar chart of peak memory by technique
- `plot_pareto_frontier()` -- memory vs quality scatter
- `plot_id_vs_ood()` -- grouped bar chart for domain transfer analysis
- `export_svg()` -- SVG export helper

### `apple_silicon.py`
- `check_mps_availability()` -- system info + MPS status
- `get_memory_stats()` -- MPS/CUDA memory usage
- `optimize_for_mps()` -- move model to MPS + set eager attention
- `mps_safe_generate()` -- generate with CPU fallback on MPS errors

---

## 7. Training Configurations (`configs/`)

Seven YAML config files provide default hyperparameters for each technique. Each includes a `low_resource` section for 6GB GPU / Colab fallback:

| Config | Model | Key Settings |
|---|---|---|
| `full_finetune.yaml` | SmolLM2-135M | lr=2e-5, batch=4, epochs=3, grad_accum=4 |
| `lora.yaml` | SmolLM2-360M | r=16, alpha=32, targets=[q,v], lr=2e-4 |
| `dora.yaml` | SmolLM2-360M | Same as LoRA with use_dora=True, targets=[q,v,k,o] |
| `qlora.yaml` | Qwen2.5-1.5B | nf4 quant, double quant, targets=[q,v,k,o] |
| `prompt_tuning.yaml` | SmolLM2-360M | 20 virtual tokens, TEXT init |
| `prefix_tuning.yaml` | SmolLM2-360M | 30 virtual tokens, prefix projection=True |
| `dpo.yaml` | SmolLM2-360M | beta=0.1, sigmoid loss, lr=5e-6, epochs=1 |

---

## 8. Documentation Site (`docs/`)

A fully custom GitHub Pages site with 10 HTML pages. No framework -- just vanilla HTML + custom CSS + vanilla JS. Key features:

### Pages
1. **index.html** -- Landing page with rotating quotes, stats, technique picker
2. **guide.html** -- 3 learning tracks (beginner/intermediate/advanced), checklist with localStorage, Mermaid prereq graph
3. **techniques.html** -- 10 tabbed technique deep-dives with SVGs, KaTeX math, code blocks, Chart.js
4. **comparison.html** -- Sortable table, VRAM filter slider, Pareto scatter, radar picker, cost calculator
5. **choosing-method.html** -- 4-question wizard, 12 task cards, Mermaid flowchart
6. **safety.html** -- Concentric risk-ring SVG, 9-card risk taxonomy, governance checklist
7. **failure-points.html** -- 10 accordion failure cards with severity filter, 5 interactive Chart.js training curves
8. **fun-usecases.html** -- 3 showcases (Shakespeare/Code Roast/Dungeon Master) with prompt picker + pre-generated outputs
9. **industry-usecases.html** -- KPI cards, Mermaid architecture diagrams, ROI calculator
10. **404.html** -- Friendly error page

### Site Infrastructure
- **CSS**: `style.css` (14KB design system), `print.css` (5KB ink-friendly print styles)
- **JS**: `main.js` (dark mode, tabs, URL-hash deep-linking, copy buttons, sortable tables, KaTeX/Mermaid init), `search.js` (modal with `/` shortcut, weighted scoring, highlighting), `search-index.js` (9-entry auto-generated index)
- **7 custom SVGs** in `docs/images/diagrams/`: taxonomy tree, LoRA architecture, DoRA vs LoRA, QLoRA flow, DPO loop, model merging, safety taxonomy rings
- **18 result PNGs** in `docs/images/notebooks/`: charts from each notebook execution

### CI/CD
- `.github/workflows/pages.yml` deploys on push to `main` (only when `docs/` changes)
- 3 pre-deploy checks: emoji audit (no emojis in source), absolute URL audit (all relative paths), search index rebuild
- Constraint: no emojis in HTML/CSS/JS/SVG; no absolute URL paths (would break at `/finetuning-slm/` sub-path)

---

## 9. Agent System

Five AI agents defined in `.github/agents/` for a phased development pipeline:

```
developer -> reviewer -> documentation + visualization -> humanizer
```

| Agent | Role |
|---|---|
| `developer` | Implements notebooks/code per plan |
| `reviewer` | Audits implementation against plan (read-only, PASS/FAIL) |
| `documentation` | Creates/updates HTML pages (fun tone, Tailwind-styled) |
| `visualization` | Creates SVGs using project color palette |
| `humanizer` | Final polish -- readable, witty, technically accurate |

---

## 10. Current Project Status

**Last updated in plan**: 2026-09-09  
**Current phase**: Phase 13 (Polish & Final Validation), Steps 27-28

### What is Done (Phases 1-12)
- All 20 notebooks built and verified end-to-end on M4 Max
- All `src/` utility modules implemented
- All 7 YAML configs created
- Full docs site (10 HTML pages, 7 custom SVGs, 18 result PNGs)
- CI/CD workflow for GitHub Pages
- 18 result PNG visualizations generated from notebook execution
- Search functionality with auto-generated index
- Dark mode, responsive design, print styles

### What Remains (Phase 13)

**Step 27 -- Documentation & Packaging:**
- [ ] Refresh `README.md` with badges, TOC, screenshots, live Pages URL
- [ ] BibTeX citation block
- [ ] `CONTRIBUTING.md`
- [ ] `LICENSE` file (MIT)
- [ ] Verify `.gitignore` completeness
- [ ] Document synthetic data provenance in `data/README.md`

**Step 28 -- Final Validation:**
- [ ] Re-run all 20 notebooks end-to-end (fresh kernel)
- [ ] Confirm memory bounds (< 35GB peak)
- [ ] Verify specific notebook behaviors (guardrail erosion in 12, fail-fix in 11, ID/OOD in 13)
- [ ] Fresh-clone test
- [ ] Cross-browser testing (Chrome, Safari, Firefox, desktop + mobile)
- [ ] First live deploy + smoke test
- [ ] README hero screenshots from deployed site

---

## 11. Key Technical Decisions & Gotchas

These are documented in the plan's "resume-tomorrow snapshot" and represent hard-won knowledge:

1. **trl 1.9.x API changes**: `SFTConfig` uses `max_length` (not `max_seq_length`). `DPOConfig` dropped `max_prompt_length`.

2. **Gradient checkpointing cleanup**: After every `SFTTrainer.train()` or `DPOTrainer.train()`, must call:
   ```python
   model.gradient_checkpointing_disable()
   model.config.use_cache = True
   model.eval()
   ```
   Otherwise `model.generate()` produces garbage.

3. **Chat template required**: SmolLM2-Instruct uses ChatML. Always use `tokenizer.apply_chat_template()` for both training data and inference.

4. **135M capacity wall**: Structural/vocabulary transfer works (notebooks 15, 17, 18). Stylistic tone transfer does not (notebooks 14, 15 roast tone, 16 DM narration). Documented honestly.

5. **MPS vs MLX**: PyTorch/MPS beats MLX at 135M (103 vs 68 tok/s). MLX advantage grows with model size.

6. **Model fallback pattern**: Every notebook tries SmolLM2-360M first, falls back to SmolLM2-135M if not cached.

7. **extract_json bug**: In notebook 17, must check `isinstance(result, dict)` before `.get()` because `json.loads("string")` returns str.

---

## 12. Dependencies

Core ML stack (from `pyproject.toml`):
- **torch >= 2.12** (MPS/CUDA backend)
- **transformers >= 5.12** (model loading, tokenizers)
- **peft >= 0.19** (LoRA, DoRA, QLoRA, Prompt/Prefix Tuning)
- **trl >= 1.6** (SFT, DPO trainers)
- **bitsandbytes >= 0.49.2** (4-bit quantization)
- **datasets >= 5.0** (HF dataset loading)
- **accelerate >= 1.14** (distributed/mixed-precision)

Evaluation: evaluate, rouge-score, bert-score, scikit-learn  
Visualization: matplotlib, plotly, seaborn  
Utilities: rich, pyyaml, jinja2, pydantic  
Inference fallback: llama-cpp-python >= 0.3.31 (GGUF)  
Optional: mlx + mlx-lm (Apple Silicon native, notebook 19)

Build system: hatchling  
Linter: ruff (line-length=100, target=py311)

---

## 13. Datasets Used

| Dataset | Source | Used In | License |
|---|---|---|---|
| databricks/dolly-15k | HF Hub | Notebooks 02, 04, 05 | CC-BY-SA-3.0 |
| tatsu-lab/alpaca | HF Hub | Notebooks 01, 04 | CC-BY-NC-4.0 |
| stanfordnlp/sst2 | HF Hub | Notebook 03 | MIT |
| Custom synthetic | Generated in-notebook | Notebooks 14-18 | Original |

No real patient data is used. Clinical notes (notebook 18) are synthetically generated. Code review data (notebook 15) is synthetically generated. The `data_utils.py` module includes local fallback datasets (240 rows each) for dolly-15k and alpaca in case HF Hub is unreachable.

---

## 14. Summary

This is a mature, nearly complete educational project with:
- **20 working notebooks** covering the full spectrum of SLM fine-tuning (full FT, LoRA, DoRA, QLoRA, prompt tuning, prefix tuning, DPO, distillation, merging, plus fun and industry use cases)
- **A polished docs site** with interactive visualizations, decision wizards, and searchable content
- **Multi-hardware support** from 6GB GPUs to Apple Silicon to cloud A100s
- **Production-quality infrastructure** (CI/CD, sanity checks, search, dark mode, print styles)

The remaining work (Phase 13) is packaging polish: README badges, LICENSE file, CONTRIBUTING.md, fresh-clone validation, cross-browser testing, and first live deployment.
