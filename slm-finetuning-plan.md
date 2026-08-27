# SLM Fine-Tuning: Comprehensive Tutorial & Implementation Plan

> **Goal**: Build an academia-ready, end-to-end Small Language Model fine-tuning tutorial covering 10+ techniques (basic → advanced, including DoRA), with 20 Jupyter notebooks, fun + novel industry use cases, dedicated safety/responsible-finetuning phase, evaluation theory depth, dataset design guidance, an emerging methods appendix, and an interactive GitHub Pages site with infographics — runnable on Apple Silicon (M1–M4), Google Colab (free tier), or any system with a 6GB+ GPU.

---

## 🚦 Implementation Progress

> **Last updated**: 2026-08-26 | **Current phase**: Phase 9, Step 19

| Phase | Step | Deliverable | Status |
|---|---|---|---|
| 1 | Step 1 | Project scaffolding (dirs, src/, configs, notebooks) | ✅ Done |
| 1 | Step 2 | `00_environment_setup.ipynb` | ✅ Done — verified on M4 Max |
| 1 | Step 3 | `01_dataset_design_and_data_prep.ipynb` | ✅ Done — verified on M4 Max |
| 2 | Step 4 | `02_full_finetuning.ipynb` | ✅ Done — pipeline verified (1-epoch run), full training deferred |
| 2 | Step 5 | `03_feature_extraction_linear_probing.ipynb` | ✅ Done — verified on M4 Max |
| 3 | Step 6 | `04_lora_and_dora_finetuning.ipynb` | ✅ Done — verified on M4 Max |
| 3 | Step 7 | `05_qlora_finetuning.ipynb` | ✅ Done — verified on M4 Max |
| 4 | Step 9 | `06_prompt_tuning.ipynb` | ✅ Done — verified on M4 Max |
| 4 | Step 10 | `07_prefix_tuning.ipynb` | ✅ Done — verified on M4 Max |
| 5 | Step 11 | `08_dpo_alignment.ipynb` | ✅ Done — verified on M4 Max |
| 5 | Step 12 | `09_knowledge_distillation.ipynb` | ✅ Done — verified on M4 Max |
| 5 | Step 13 | `10_model_merging.ipynb` | ✅ Done — verified on M4 Max |
| 6 | Step 14 | `12_responsible_safe_finetuning.ipynb` | 🔜 Next |
| 7 | Step 15 | `11_failure_points_and_debugging.ipynb` | ✅ Done — verified on M4 Max |
| 8 | Step 16 | `13_evaluation_theory_and_benchmarking.ipynb` | ✅ Done — verified on M4 Max |
| 9 | Steps 17–19 | Notebooks 14–16 (fun use cases) | 🔜 In progress (14, 15 done) |
| 10 | Steps 20–21 | Notebooks 17–18 (industry use cases) | ⬜ Not started |
| 11 | Step 22 | `19_apple_silicon_optimization_guide.ipynb` | ⬜ Not started |
| 12 | Steps 23–26 | docs/ HTML site (9 pages) | ⬜ Not started |
| 13 | Steps 27–28 | Polish & final validation | ⬜ Not started |

### How to Continue Development

**Agents available** (defined in `.github/agents/` and `AGENTS.md`):

| Agent | Role | Invocation |
|---|---|---|
| `developer` | Implements notebooks/code per this plan | Primary worker |
| `reviewer` | Audits implementation vs plan (read-only) | After each implementation |
| `documentation` | Creates/updates HTML pages with fun tone | After reviewer approves |
| `visualization` | Creates SVG diagrams for docs/images/ | Called by documentation agent |
| `humanizer` | Final humor/engagement polish on docs | Last step before publish |

**Pipeline per step**: Developer → Reviewer → (fix gaps) → Documentation + Visualization → Humanizer → Commit

**Next action**: Build `notebooks/16_fun_dungeon_master.ipynb` following Phase 9 Step 19 below.

---

## Table of Contents

1. [Target Models](#target-models)
2. [Hardware & Environment](#hardware--environment)
3. [Project Structure](#project-structure)
4. [Choosing a Fine-Tuning Method — Task-to-Technique Guide](#choosing-a-fine-tuning-method)
5. [Phase 1 — Foundation & Infrastructure](#phase-1--foundation--infrastructure)
6. [Phase 2 — Basic Techniques](#phase-2--basic-techniques)
7. [Phase 3 — Parameter-Efficient Techniques (Core)](#phase-3--parameter-efficient-techniques-core)
8. [Phase 4 — Prompt-Based Techniques](#phase-4--prompt-based-techniques)
9. [Phase 5 — Alignment & Advanced Techniques](#phase-5--alignment--advanced-techniques)
10. [Phase 6 — Responsible & Safe Fine-Tuning](#phase-6--responsible--safe-fine-tuning)
11. [Phase 7 — Failure Analysis & Debugging](#phase-7--failure-analysis--debugging)
12. [Phase 8 — Evaluation Theory & Benchmarking](#phase-8--evaluation-theory--benchmarking)
13. [Phase 9 — Fun Use Cases](#phase-9--fun-use-cases)
14. [Phase 10 — Novel Industry Use Cases](#phase-10--novel-industry-use-cases)
15. [Phase 11 — Apple Silicon Optimization](#phase-11--apple-silicon-optimization)
16. [Phase 12 — Interactive GitHub Pages Site](#phase-12--interactive-github-pages-site)
17. [Phase 13 — Polish & Academic Readiness](#phase-13--polish--academic-readiness)
18. [Dependency Graph](#dependency-graph)
19. [Verification Checklist](#verification-checklist)
20. [Design Decisions](#design-decisions)
21. [Appendix A — Emerging Methods](#appendix-a--emerging-methods)

---

## Target Models

### Primary Models (Open — No Token Required)

| Model | Parameters | Role | Rationale |
|---|---|---|---|
| **SmolLM2-135M** | 135M | Quick demos, distillation student, 6GB GPU friendly | Instant iteration; fits on any hardware |
| **SmolLM2-360M** | 360M | Main tutorials | Best speed/capability balance for teaching |
| **Qwen2.5-1.5B-Instruct** | 1.5B | Quality-critical demos | Better reasoning for code review, clinical tasks |
| **Qwen2.5-3B-Instruct** | 3B | Distillation teacher | Large enough to teach, small enough for 36GB+ systems |

### Alternative Models (Gated — HuggingFace Token Required)

| Model | Parameters | License | Why Include |
|---|---|---|---|
| **Llama 3.2 1B** | 1B | Llama 3.2 Community | Strong quality/size ratio; popular in industry |
| **Llama 3.2 3B** | 3B | Llama 3.2 Community | Best reasoning at 3B scale |
| **Gemma 2 2B** | 2B | Gemma License | Excellent instruction following; Google ecosystem |

> **Note**: The tutorial works end-to-end with open models only. Gated models are optional alternatives — students who accept the license on HuggingFace can swap them in for any notebook. See README for HF token setup instructions.

---

## Hardware & Environment

### Supported Hardware Tiers

| Tier | Hardware | What Works |
|---|---|---|
| **6 GB GPU** | RTX 3060, GTX 1660, Colab T4 (free) | SmolLM2-135M (all techniques), 360M/1.5B (QLoRA only) |
| **16 GB** | Apple M1/M2 (16GB), RTX 4060 Ti | SmolLM2-360M (LoRA/DoRA), Qwen2.5-1.5B (QLoRA) |
| **36 GB** | Apple M3/M4 (36GB), RTX 3090 | All notebooks including Qwen2.5-1.5B LoRA |
| **64+ GB** | Apple M-series (64GB+), A100 | Everything at full precision incl. Qwen2.5-3B distillation |

> **Design principle**: SmolLM2-135M runs ALL techniques even on 6GB, so no student is locked out of any concept. Each notebook includes a "low-resource" config fallback.

### Apple M4 Max Specifics (Primary Development Machine)

- **PyTorch MPS backend** for GPU acceleration via Metal Performance Shaders
- **Unified memory architecture** (up to 128GB) — significant advantage over discrete GPU setups for loading larger models
- `bitsandbytes >=0.49.2` (already installed) has Apple Silicon 4-bit quantization support — verify at runtime; fallback to fp16 LoRA if unstable
- MLX framework available as alternative for native Apple Silicon fine-tuning

### Google Colab Compatibility

- Each notebook includes a Colab setup cell at the top (`!pip install ...`)
- Runtime: T4 GPU (15GB VRAM) — sufficient for all techniques up to 3B QLoRA
- Students mount Google Drive to persist checkpoints across sessions
- Session limit: ~12 hours — notebooks save checkpoints at each epoch

### Python Environment (Local)

```
Primary:  ~/python-envs/agentic-slm/   (Python 3.11)
Alternative: project-local .venv via `uv sync` (see setup.sh)
```

**Already installed**: `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, `evaluate`, `rouge-score`, `bert-score`, `safetensors`, `tokenizers`, `scikit-learn`

**To add**: `plotly` (interactive charts)

```bash
cd ~/python-envs/agentic-slm && UV_SYSTEM_CERTS=1 uv add plotly
```

---

## Project Structure

```
finetuning-slm/
├── README.md                              # Project overview, quickstart, table of contents
├── slm-finetuning-plan.md                 # This plan document
│
├── docs/                                  # ── GitHub Pages Site ──
│   ├── index.html                         # Landing page + learning path navigation
│   ├── techniques.html                    # All techniques explained with visuals
│   ├── comparison.html                    # Interactive sortable comparison table + charts
│   ├── choosing-method.html               # Task-to-technique decision guide
│   ├── failure-points.html                # Visual catalog of failure modes
│   ├── safety.html                        # Responsible fine-tuning guide
│   ├── fun-usecases.html                  # Fun use case showcase
│   ├── industry-usecases.html             # Novel industry use case deep-dives
│   ├── guide.html                         # Learning curriculum
│   ├── css/
│   │   └── style.css                      # Tailwind (CDN) + custom styles
│   ├── js/
│   │   └── main.js                        # Interactivity (sorting, tabs, dark mode)
│   └── images/                            # SVG/PNG infographics
│       ├── technique-taxonomy.svg         # Tree of all fine-tuning methods
│       ├── task-technique-matrix.svg      # Task → technique mapping visual
│       ├── lora-architecture.svg          # LoRA low-rank decomposition visual
│       ├── dora-architecture.svg          # DoRA magnitude/direction decomposition
│       ├── qlora-architecture.svg         # QLoRA quantization + LoRA visual
│       ├── prompt-tuning-visual.svg       # Soft prompt prepending visual
│       ├── prefix-tuning-visual.svg       # Layer-wise prefix visual
│       ├── dpo-vs-rlhf.svg               # DPO vs RLHF comparison diagram
│       ├── distillation-pipeline.svg      # Teacher-student framework
│       ├── safety-risk-taxonomy.svg       # Risks from fine-tuning visual
│       ├── dataset-design-flowchart.svg   # Dataset construction decision guide
│       ├── evaluation-limitations.svg     # Metric pitfalls infographic
│       ├── decision-flowchart.svg         # "Which technique?" interactive guide
│       └── memory-comparison.svg          # Memory usage across techniques
│
├── notebooks/                             # ── 20 Jupyter Notebooks ──
│   ├── 00_environment_setup.ipynb
│   ├── 01_dataset_design_and_data_prep.ipynb
│   ├── 02_full_finetuning.ipynb
│   ├── 03_feature_extraction_linear_probing.ipynb
│   ├── 04_lora_and_dora_finetuning.ipynb           # LoRA + DoRA comparison
│   ├── 05_qlora_finetuning.ipynb
│   ├── 06_prompt_tuning.ipynb
│   ├── 07_prefix_tuning.ipynb
│   ├── 08_dpo_alignment.ipynb
│   ├── 09_knowledge_distillation.ipynb
│   ├── 10_model_merging.ipynb
│   ├── 11_failure_points_and_debugging.ipynb
│   ├── 12_responsible_safe_finetuning.ipynb
│   ├── 13_evaluation_theory_and_benchmarking.ipynb
│   ├── 14_fun_shakespearean_insult_generator.ipynb
│   ├── 15_fun_code_roast_reviewer.ipynb
│   ├── 16_fun_dungeon_master.ipynb
│   ├── 17_industry_agentic_tool_calling.ipynb
│   ├── 18_industry_privacy_preserving_clinical_summarizer.ipynb
│   ├── 19_apple_silicon_optimization_guide.ipynb
│   └── README.md                          # Notebook index with descriptions
│
├── src/                                   # ── Reusable Python Utilities ──
│   ├── __init__.py
│   ├── data_utils.py                      # Dataset loading, formatting, chat templates
│   ├── training_utils.py                  # Training helpers, callbacks, MPS setup
│   ├── evaluation_utils.py                # ROUGE, BERTScore, perplexity, JSON validity
│   ├── safety_utils.py                    # Red-teaming prompts, toxicity checks
│   ├── visualization.py                   # Comparison charts, training curves, export SVG
│   └── apple_silicon.py                   # MPS device management, memory monitoring
│
├── configs/                               # ── YAML Training Configs ──
│   ├── full_finetune.yaml
│   ├── lora.yaml
│   ├── dora.yaml
│   ├── qlora.yaml
│   ├── prompt_tuning.yaml
│   ├── prefix_tuning.yaml
│   └── dpo.yaml
│
├── data/
│   ├── .gitkeep
│   └── README.md                          # Data sourcing, licensing, governance notes
│
├── outputs/
│   └── .gitkeep
│
└── .github/
    └── workflows/
        └── pages.yml                      # GitHub Pages deployment workflow
```

---

## Choosing a Fine-Tuning Method

> A task-centric decision guide that maps real-world objectives to specific techniques with reasoning. This section is both a conceptual framework taught in the notebooks AND a dedicated HTML page (`choosing-method.html`).

### Task → Technique Mapping

| Task Objective | Primary Technique | Alternative | Rationale |
|---|---|---|---|
| **Instruction following** | LoRA / DoRA | QLoRA (if memory-limited) | Modifies attention weights to follow instruction patterns; DoRA improves directional learning |
| **Domain adaptation** (medical, legal, finance) | LoRA with higher rank (r=32–64) | Full FT (if model is small enough) | Needs deeper knowledge injection; higher rank captures more domain signal |
| **Style transfer** (tone, persona, writing style) | LoRA (r=8–16) | Prompt Tuning (if style is simple) | Style is a relatively "thin" transformation — low rank suffices |
| **Alignment / safety** | DPO (on LoRA-tuned base) | ORPO (emerging, single-stage) | Steers behavior without reward model; DPO is stable and well-supported |
| **Structured output / tool-calling** | LoRA with constrained decoding | Full FT (if small model) | Needs reliable format adherence; LoRA + format-focused data is sufficient |
| **Classification / extraction** | Feature Extraction or Prompt Tuning | LoRA (if complex) | Simple output space; lightweight methods work |
| **Knowledge compression** | Knowledge Distillation | Distillation + LoRA on student | Transfers capability from large to small model |
| **Multi-task without retraining** | Model Merging (TIES/DARE) | Multi-task LoRA | Combines existing adapters zero-shot |
| **Maximum quality, no constraints** | Full Fine-Tuning | DoRA (near-full-FT quality at LoRA cost) | Updates all parameters for maximum adaptation |
| **Edge/on-device deployment** | QLoRA + GGUF export | Distillation → small model | Memory-constrained environments need quantization |

### Decision Flowchart Logic

```
START: What is your primary constraint?
│
├─► Memory limited? ──────────► QLoRA (large model) or Prompt Tuning (small adaptation)
│
├─► Compute limited? ─────────► LoRA (few epochs) or Prompt Tuning
│
├─► Need alignment? ──────────► DPO on LoRA-tuned base
│
├─► Multiple tasks? ──────────► Train separate LoRA adapters → Merge (TIES/DARE)
│
├─► Need tiny model? ─────────► Knowledge Distillation from larger teacher
│
├─► Maximum quality? ─────────► Full FT (small model) or DoRA (larger model)
│
└─► Unsure? ──────────────────► Start with LoRA (r=16, alpha=32) — it's the safe default
```

### Method Properties at a Glance

| Method | Trainable % | Memory | Quality | Speed | Complexity |
|---|---|---|---|---|---|
| Full FT | 100% | ●●●●● | ●●●●● | ●●○○○ | ●○○○○ |
| LoRA | 0.1–1% | ●●○○○ | ●●●●○ | ●●●●○ | ●●○○○ |
| **DoRA** | 0.1–1% | ●●○○○ | ●●●●◐ | ●●●○○ | ●●○○○ |
| QLoRA | 0.1–1% | ●○○○○ | ●●●○○ | ●●●○○ | ●●●○○ |
| Prompt Tuning | <0.01% | ●○○○○ | ●●○○○ | ●●●●● | ●○○○○ |
| Prefix Tuning | ~0.1% | ●○○○○ | ●●●○○ | ●●●●○ | ●●○○○ |
| DPO | (LoRA %) | ●●○○○ | N/A (alignment) | ●●●○○ | ●●●○○ |
| Distillation | 100% (student) | ●●●○○ | ●●●○○ | ●●○○○ | ●●●●○ |
| Merging | 0% (inference) | ●○○○○ | ●●●○○ | ●●●●● | ●●○○○ |

> **Key insight for practitioners**: LoRA is the correct default for 80% of fine-tuning tasks. DoRA is the upgrade when you need that last 5–10% quality improvement without the cost of full fine-tuning. Start with LoRA, escalate only when results demand it.

---

## Phase 1 — Foundation & Infrastructure

> **Steps 1–3** · Blocks all subsequent phases

### Step 1: Project Scaffolding

- Create the full directory structure above
- Write comprehensive `README.md` with project overview, learning path, prerequisites
- Create `src/` utility modules:
  - `data_utils.py` → `load_and_format_dataset()`, `apply_chat_template()`, `create_preference_pairs()`, `generate_synthetic_data()`, `compute_dataset_statistics()`, `check_data_quality()`
  - `training_utils.py` → `get_training_args()`, `setup_mps_device()`, `MemoryMonitorCallback`, `TrainingMetricsCallback`
  - `evaluation_utils.py` → `compute_rouge()`, `compute_bertscore()`, `compute_perplexity()`, `evaluate_json_validity()`, `evaluate_tool_call_accuracy()`, `compute_ood_degradation()`
  - `safety_utils.py` → `red_team_prompts()`, `check_toxicity()`, `test_guardrail_retention()`, `measure_refusal_rate()`
  - `visualization.py` → `plot_training_curves()`, `plot_technique_comparison_radar()`, `plot_memory_comparison()`, `plot_pareto_frontier()`, `plot_id_vs_ood()`, `export_svg()`
  - `apple_silicon.py` → `check_mps_availability()`, `get_memory_stats()`, `optimize_for_mps()`, `mps_workaround_decorator()`
- Create YAML configs:

  | Config | Key Parameters |
  |---|---|
  | `full_finetune.yaml` | epochs=3, lr=2e-5, batch_size=4, grad_accum=4 |
  | `lora.yaml` | r=16, alpha=32, targets=["q_proj","v_proj"], dropout=0.05 |
  | `dora.yaml` | r=16, alpha=32, use_dora=True, targets=["q_proj","v_proj","k_proj","o_proj"] |
  | `qlora.yaml` | Same as LoRA + bnb_4bit_quant_type="nf4", compute_dtype=bfloat16 |
  | `prompt_tuning.yaml` | num_virtual_tokens=20, init="TEXT" |
  | `prefix_tuning.yaml` | num_virtual_tokens=30, prefix_projection=True |
  | `dpo.yaml` | beta=0.1, lr=5e-6, max_length=512 |

### Step 2: Environment Setup Notebook (`00_environment_setup.ipynb`)

- Verify `agentic-slm` kernel, all package versions
- Detect MPS availability and print device info
- Install any missing packages (`matplotlib`, `plotly`)
- Download SmolLM2-135M and run a quick inference test
- Memory profiling: print available unified memory, MPS allocatable memory
- **Benchmark**: CPU vs MPS inference speed comparison (table + chart)
- Quick orientation: summarize the "Choosing a Fine-Tuning Method" guide so readers know what's ahead

### Step 3: Dataset Design & Data Preparation (`01_dataset_design_and_data_prep.ipynb`)

> Goes beyond "load from HuggingFace" to teach HOW to design and construct a fine-tuning dataset for a given objective.

#### Part A: Dataset Design Principles (Conceptual)

- **Objective-driven corpus construction**: How the fine-tuning objective determines data requirements
  - Instruction tuning → needs diverse instruction/response pairs with varied complexity
  - Domain adaptation → needs domain-specific text at volume (quality > diversity)
  - Style transfer → needs parallel examples in target style (small but high-quality)
  - Alignment → needs preference pairs with clear quality delta
  - Tool-calling → needs schema-grounded examples with edge cases
- **Data quality hierarchy**: Why 1,000 excellent examples beat 100,000 noisy ones
  - Demonstrate empirically: train on 500 clean vs 5,000 noisy examples → compare
- **Volume guidelines by technique**:

  | Technique | Minimum Viable Dataset | Sweet Spot | Diminishing Returns |
  |---|---|---|---|
  | Full FT | 5,000 | 50,000–100,000 | >500K |
  | LoRA / DoRA | 1,000 | 5,000–20,000 | >100K |
  | QLoRA | 1,000 | 5,000–20,000 | >100K |
  | Prompt Tuning | 500 | 2,000–5,000 | >20K |
  | DPO | 1,000 pairs | 5,000–10,000 pairs | >50K pairs |
  | Distillation | 10,000 | 50,000–200,000 | >1M |

#### Part B: Labeling Strategies

- **Human annotation**: When and how — quality control, inter-annotator agreement, annotation guidelines design
- **Synthetic data generation**: Using larger models to generate training data for smaller ones
  - Trade-offs: cost, diversity, potential bias inheritance, "model collapse" risk
  - Filtering synthetic data: perplexity-based, rule-based, model-as-judge
- **Self-instruct / Evol-Instruct patterns**: Programmatic complexity escalation
- **Hybrid approaches**: Seed with human data, expand with synthetic, filter with human validation
- **When NOT to use synthetic data**: domains where factual accuracy is critical (medical, legal)

#### Part C: Data Governance & Licensing

- **Licensing considerations**: MIT, Apache 2.0, CC-BY vs CC-BY-NC, proprietary data risks
  - Code datasets: check license compatibility (e.g., GitHub Copilot controversy)
  - Medical data: HIPAA de-identification requirements, synthetic alternatives
  - Web-scraped data: robots.txt, copyright, opt-out mechanisms
- **Privacy**: PII detection and scrubbing before fine-tuning
- **Reproducibility**: Dataset versioning, data cards, documentation requirements
- **Bias audit**: Check for demographic imbalance, toxicity, stereotypes in training data

#### Part D: Practical Implementation

- Deep dive into tokenizer types (BPE, WordPiece, SentencePiece, Unigram) with visual comparison
- Chat template formatting: ChatML, Alpaca, Llama-style, custom formats
- Load datasets from HuggingFace Hub: `databricks/dolly-15k`, `tatsu-lab/alpaca`
- Data quality checks: length distribution, deduplication, format validation
- Train/val/test splitting strategies for fine-tuning
- 🔴 **Failure Point**: Wrong chat template → garbled model output (demonstrate + fix)
- 🔴 **Failure Point**: PII leakage in training data → model memorizes sensitive information

---

## Phase 2 — Basic Techniques

> **Steps 4–5** · Depends on Phase 1

### Step 4: Full Fine-Tuning (`02_full_finetuning.ipynb`)

**Concept**: Update ALL model parameters on a downstream task.

- Fine-tune SmolLM2-135M on a subset of `databricks/dolly-15k` for instruction following
- Track: memory consumption, training time, total trainable parameter count
- Compare before/after generation quality (side-by-side output table)
- Save and reload the fine-tuned model
- 🔴 **Failure Point #1**: Catastrophic forgetting — show the model losing general knowledge after aggressive fine-tuning (demonstrate with pre/post general Q&A)
- 🔴 **Failure Point #2**: Overfitting on small dataset — show loss curve going down while generation quality degrades, wrong learning rate causes divergence

### Step 5: Feature Extraction / Linear Probing (`03_feature_extraction_linear_probing.ipynb`)

**Concept**: Freeze all base model parameters, train only a new classification head.

- Use SmolLM2-135M as a frozen feature extractor
- Add classification head for sentiment analysis (SST-2 or similar)
- Show: extremely fast training (~minutes), but limited to classification tasks
- Comparison table vs full fine-tuning: accuracy, speed, memory, parameter count

---

## Phase 3 — Parameter-Efficient Techniques (Core)

> **Steps 6–8** · Depends on Phase 1 · This is the heart of the tutorial

### Step 6: LoRA & DoRA Fine-Tuning (`04_lora_and_dora_finetuning.ipynb`) ★ Flagship

#### Part A: LoRA (Low-Rank Adaptation)

**Concept**: Decompose weight updates into low-rank matrices W = W₀ + BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×d), and r ≪ d.

- **Visual explanation**: Inline matplotlib diagram showing low-rank decomposition
- Hyperparameter deep-dive with experiments:

  | Experiment | Rank (r) | Alpha | Target Modules | Trainable % |
  |---|---|---|---|---|
  | Minimal | 4 | 8 | q_proj, v_proj | ~0.05% |
  | Standard | 16 | 32 | q_proj, v_proj | ~0.19% |
  | Aggressive | 64 | 128 | all linear | ~0.8% |

- Fine-tune SmolLM2-360M on instruction-following dataset
- Side-by-side: trainable parameter count vs full fine-tuning
- Adapter merging into base model and export
- Memory and speed comparison with full fine-tuning
- 🔴 **Failure Point #1**: Wrong target modules → model doesn't learn (flat loss curve)
- 🔴 **Failure Point #2**: Rank too low → underfitting; too high → overfitting (comparative training curves)

#### Part B: DoRA (Weight-Decomposed Low-Rank Adaptation)

**Concept**: Decomposes pre-trained weight into **magnitude** (m) and **direction** (V) components, then applies LoRA only to the directional component:

W' = m · (V + ΔV) / ‖V + ΔV‖_c

where ΔV = BA is the LoRA update applied to direction only.

- **Visual**: Side-by-side diagram of LoRA vs DoRA weight decomposition
- **Why DoRA matters**: Bridges the quality gap between LoRA and full fine-tuning by learning magnitude and direction independently — matches the learning pattern of full FT
- Implement using PEFT's `use_dora=True` parameter (supported since PEFT v0.9+)
- **Head-to-head comparison on same task**:

  | Metric | LoRA (r=16) | DoRA (r=16) | Full FT | Notes |
  |---|---|---|---|---|
  | Eval Loss | baseline | ~3–8% lower | best | DoRA closes the gap |
  | Trainable % | 0.19% | 0.19% + magnitude | 100% | Negligible overhead |
  | Training speed | fast | ~10% slower | slow | Extra norm computation |
  | Memory | low | low (+<5%) | high | Magnitude vectors are small |

- When to choose DoRA over LoRA:
  - Domain adaptation where quality is critical
  - Tasks where LoRA plateaus and full FT is too expensive
  - Models where directional learning matters (observed in vision + NLP)
- When LoRA is sufficient:
  - Style transfer, simple instruction tuning, prototyping
  - Maximum speed / iteration velocity needed

#### Part C: LoRA vs DoRA vs Full FT Unified Comparison

- Training curves overlay (loss vs steps for all three)
- Quality-memory Pareto chart
- Recommendation: "Start with LoRA. If quality plateau → switch to DoRA. Full FT only if model is tiny."

### Step 7: QLoRA Fine-Tuning (`05_qlora_finetuning.ipynb`)

**Concept**: Quantize base model to 4-bit, apply LoRA adapters on top. Enables training models that wouldn't fit in memory otherwise.

- Visual explanation of 4-bit NormalFloat (NF4) quantization
- Load SmolLM2-360M in 4-bit with `bitsandbytes`
- Apply LoRA adapters on quantized model
- **Memory comparison chart** (bar chart):

  ```
  Full FT (360M):   ~3.2 GB
  LoRA (360M):      ~2.1 GB
  DoRA (360M):      ~2.2 GB
  QLoRA (360M):     ~1.1 GB
  QLoRA (1.5B):     ~2.8 GB  ← wouldn't fit with Full FT!
  ```

- Demonstrate training Qwen2.5-1.5B via QLoRA (impossible with full FT on same hardware budget)
- 🔴 **Failure Point**: Quantization accuracy degradation — show perplexity increase from quantization, compare 4-bit vs 8-bit vs fp16
- **Apple Silicon note**: If `bitsandbytes` 4-bit is unstable on MPS, document the issue and demonstrate GGUF alternative via `llama-cpp-python`

### Step 8: LoRA vs DoRA vs QLoRA vs Full FT Comparison (embedded in Notebooks 04–05)

- Training curves overlay (loss vs steps for all four)
- Memory/speed/quality Pareto chart
- Summary recommendation table with decision criteria

---

## Phase 4 — Prompt-Based Techniques

> **Steps 9–10** · Depends on Phase 1 · Can run parallel with Phases 2–3

### Step 9: Prompt Tuning (`06_prompt_tuning.ipynb`)

**Concept**: Prepend learnable "soft prompt" embeddings to the input. Only these embeddings are trained — the model is completely frozen.

- Visual: show soft prompt tokens being prepended to input embeddings
- Implement using PEFT `PromptTuningConfig`
- Show extremely small trainable parameter count (~0.001% of model)
- Fine-tune SmolLM2-360M for a specific task (e.g., sentiment, summarization)
- 🔴 **Failure Point**: Prompt length too short (num_virtual_tokens=2) → poor performance; demonstrate and find optimal length
- Comparison with LoRA on same task: trade-off between parameter efficiency and quality

### Step 10: Prefix Tuning (`07_prefix_tuning.ipynb`)

**Concept**: Like prompt tuning, but learnable prefixes are added at EVERY transformer layer (not just input). More expressive but more parameters.

- Visual: show prefix vectors at each transformer layer vs. prompt tuning (input only)
- Implement using PEFT `PrefixTuningConfig`
- Compare with prompt tuning: trainable params, performance, inference overhead
- 🔴 **Failure Point**: Instability with very large prefix lengths — show training divergence
- Summary: Prompt Tuning vs Prefix Tuning — when to use which

---

## Phase 5 — Alignment & Advanced Techniques

> **Steps 11–13** · Depends on Phase 3 (needs LoRA knowledge)

### Step 11: DPO Alignment (`08_dpo_alignment.ipynb`)

**Concept**: Direct Preference Optimization — align model behavior using preference pairs (chosen/rejected) WITHOUT needing a separate reward model (unlike RLHF).

The DPO loss minimizes:
L_DPO = -E[log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]

- **Visual**: Side-by-side comparison of RLHF pipeline (4 models) vs DPO pipeline (2 models)
- Prepare preference dataset with chosen/rejected response pairs
- Use TRL's `DPOTrainer` with SmolLM2-360M + LoRA adapters
- Show how DPO shifts probability mass toward preferred responses
- Evaluate with win-rate comparison (before/after DPO)
- 🔴 **Failure Point #1**: Poor preference data quality → model doesn't improve or degrades
- 🔴 **Failure Point #2**: Beta too high → too conservative; beta too low → reward hacking

### Step 12: Knowledge Distillation (`09_knowledge_distillation.ipynb`)

**Concept**: Transfer knowledge from a large "teacher" model to a small "student" model using soft labels (probability distributions) rather than hard labels.

L = α · L_task(y, ŷ_student) + (1-α) · T² · KL(p_teacher^T ‖ p_student^T)

- **Visual**: Teacher-student pipeline diagram
- Teacher: Qwen2.5-3B-Instruct → Student: SmolLM2-135M
- Implement KL divergence loss + task-specific loss
- Show student approaching teacher performance at 1/22 the parameter count
- Compare: distilled student vs directly fine-tuned student (same data, same epochs)
- 🔴 **Failure Point**: Temperature tuning — T too high → uniform soft labels (no signal); T too low → equivalent to hard labels (no benefit)

### Step 13: Model Merging (`10_model_merging.ipynb`)

**Concept**: Combine multiple fine-tuned models/adapters into a single model WITHOUT additional training.

- Explain merging techniques with visuals:
  - **Linear interpolation**: W_merged = λ·W_A + (1-λ)·W_B
  - **TIES** (Trim, Elect Sign, Merge): prune small deltas, resolve sign conflicts
  - **DARE** (Drop And REscale): randomly drop delta parameters, rescale survivors
  - **SLERP** (Spherical Linear Interpolation): interpolate on hypersphere
- Train two LoRA adapters on different tasks (e.g., summarization + Q&A)
- Merge them into a single multi-task model
- Evaluate merged model on BOTH tasks
- 🔴 **Failure Point**: Merging incompatible adapters (different base models or wildly different tasks) → quality collapse on both tasks

---

## Phase 6 — Responsible & Safe Fine-Tuning

> **Step 14** · NEW PHASE · Depends on Phases 3 & 5 · Critical for industry and academic credibility

### Step 14: Responsible Fine-Tuning Notebook (`12_responsible_safe_finetuning.ipynb`)

> Fine-tuning can BREAK safety properties. This is especially critical for the agentic tool-calling and clinical use cases. This notebook teaches practitioners to recognize, measure, and mitigate safety risks introduced by fine-tuning.

#### Part A: Risk Taxonomy of Fine-Tuning

| Risk | What Happens | Why It Matters | Most Vulnerable Technique |
|---|---|---|---|
| **Jailbreak vulnerability** | Fine-tuning on unsafe data removes model's refusal training | Model will comply with harmful requests | Full FT, large LoRA rank |
| **Guardrail erosion** | Safety-tuned model loses alignment after domain fine-tuning | Even benign data can undo safety training | Full FT > LoRA > QLoRA |
| **Bias amplification** | Biased training data amplifies stereotypes in outputs | Discriminatory outputs in production | All techniques |
| **Hallucination in sensitive domains** | Model generates plausible but false information | Medical, legal, financial harm | All generative tasks |
| **Data poisoning** | Malicious examples in training set → backdoor behavior | Compromised model integrity | All techniques |
| **Privacy leakage** | Model memorizes and regurgitates training data PII | Regulatory violations (GDPR, HIPAA) | Full FT, large models |

#### Part B: Measuring Safety Before and After Fine-Tuning

- **Guardrail retention test**: Run a standard set of harmful prompts against base model vs fine-tuned model → measure refusal rate delta
  - Demonstrate: fine-tuning on benign instruction data can reduce refusal rate by 20–40%
- **Bias benchmark**: Use WinoBias or custom probes to measure gender/demographic bias amplification
- **Hallucination rate**: For factual tasks, measure % of claims not grounded in source material
- **Toxicity regression**: Use Perspective API or local toxicity classifier to measure output toxicity shift

#### Part C: Mitigation Strategies

1. **Data curation as first defense**:
   - Include 5–10% safety examples in training data (refusals, boundary cases)
   - Filter toxic/harmful content from training corpus
   - Balance demographic representation
2. **DPO for safety alignment** (tie to notebook 08):
   - After capability fine-tuning, run DPO with safety preference pairs
   - Chosen: helpful + safe response; Rejected: helpful but unsafe response
3. **Gradient masking / selective fine-tuning**:
   - Don't fine-tune safety-critical layers (if identifiable)
   - Use lower LoRA rank for layers that encode safety behavior
4. **Red-teaming protocol**:
   - Systematic adversarial testing before deployment
   - Categories: direct harm requests, indirect/social engineering, edge cases
   - Build reusable red-team prompt set in `src/safety_utils.py`

#### Part D: Practical Demo

- Take SmolLM2-360M-Instruct (safety-tuned base)
- Fine-tune with LoRA on benign instruction data
- **Measure guardrail erosion**: before/after refusal rates
- Apply mitigation: add 5% safety data to training mix → re-fine-tune
- Show refusal rate recovers without quality degradation
- Connect to clinical use case: why DPO stage 2 is essential for hallucination reduction

#### Part E: Governance Checklist for Deployment

- [ ] Safety evaluation completed (red-teaming, bias, toxicity)
- [ ] Training data audited for PII, licensing, bias
- [ ] Guardrail retention verified (refusal rate within acceptable delta)
- [ ] Domain-specific risks documented (medical: hallucination, legal: confidentiality)
- [ ] Monitoring plan for production drift

---

## Phase 7 — Failure Analysis & Debugging

> **Step 15** · Depends on Phases 2–5

### Step 15: Comprehensive Failure Points Notebook (`11_failure_points_and_debugging.ipynb`)

Deep-dive into **10 failure modes** — each with: **Demonstrate → Diagnose → Fix → Verify**

| # | Failure Mode | Severity | Technique Affected |
|---|---|---|---|
| 1 | **Catastrophic forgetting** | 🔴 Critical | Full FT |
| 2 | **Overfitting on small data** | 🔴 Critical | All |
| 3 | **Learning rate disasters** | 🟠 High | All |
| 4 | **Tokenizer mismatch** | 🔴 Critical | All |
| 5 | **Data contamination** | 🟠 High | All |
| 6 | **Evaluation metric traps** | 🟡 Medium | All |
| 7 | **MPS/Apple Silicon gotchas** | 🟠 High | All (on Mac) |
| 8 | **Quantization accuracy loss** | 🟡 Medium | QLoRA |
| 9 | **Chat template errors** | 🔴 Critical | Instruction tuning |
| 10 | **Gradient accumulation math** | 🟡 Medium | All |

Each section includes:
- **What it looks like** (symptoms, misleading metrics)
- **Live demonstration** (code that produces the failure)
- **Root cause analysis** (why it happens, with diagram)
- **The fix** (code changes, hyperparameter adjustments)
- **Verification** (proof the fix works)

---

## Phase 8 — Evaluation Theory & Benchmarking

> **Step 16** · Depends on Phases 2–5 (needs outputs from all techniques)

### Step 16: Evaluation Theory & Benchmarking (`13_evaluation_theory_and_benchmarking.ipynb`)

> Goes beyond "compute ROUGE and make a bar chart" to teach WHY metrics can mislead, HOW to evaluate generalization, and WHAT to measure beyond standard automatic metrics.

#### Part A: Evaluation Theory — Why Metrics Lie

- **Perplexity limitations**:
  - Low perplexity ≠ good generation quality (demonstrate: overfitted model has low perplexity but terrible outputs)
  - Perplexity is distribution-dependent — only meaningful relative to same test distribution
  - Cannot compare perplexity across different tokenizers/models
- **ROUGE / BLEU pitfalls**:
  - High ROUGE with factually wrong outputs (demonstrate: model copies source structure but invents facts)
  - n-gram overlap rewards verbosity and penalizes concise but correct answers
  - Demonstrate: generate a clearly bad summary that scores high ROUGE
- **BERTScore limitations**:
  - Sensitive to surface form similarity, can miss semantic errors
  - Threshold for "good" varies by domain
- **The "Goodhart's Law" of NLP metrics**: When a metric becomes a target, it ceases to be a good metric
  - Practical implication: always use 2–3 complementary metrics + qualitative inspection

#### Part B: In-Domain vs Out-of-Domain Generalization

- **Experimental design**:
  - Fine-tune SmolLM2-360M with LoRA on medical Q&A dataset
  - Evaluate on: (1) held-out medical Q&A (in-domain), (2) general Q&A (out-of-domain), (3) adjacent domain (scientific Q&A)
- **Demonstrate generalization patterns**:

  | Evaluation Set | Base Model | Fine-tuned | Delta |
  |---|---|---|---|
  | In-domain (medical) | baseline | +25% | Expected improvement |
  | Out-of-domain (general) | baseline | -5% to -15% | Slight degradation |
  | Adjacent (scientific) | baseline | +5% to +10% | Partial transfer |

- **Insight**: Fine-tuning narrows the model's capability — the "specialization tax"
- **Mitigation strategies**: Data diversity, regularization, replay buffers, DoRA (better OOD retention than LoRA)

#### Part C: Beyond Automatic Metrics

- **Model-as-judge**: Use a larger model (Qwen2.5-3B) to evaluate smaller fine-tuned model outputs
  - Design evaluation rubrics: helpfulness, accuracy, safety, relevance
  - Demonstrate inter-judge agreement between model-as-judge and human eval
- **Human evaluation proxies**: When you can't do human eval at scale
  - A/B preference testing (simplified DPO-style scoring)
  - Task completion rate (for tool-calling: did the action succeed?)
  - Factual consistency scoring (NLI-based)
- **Behavioral testing**: CheckList-style tests (capabilities, invariances, directional expectations)

#### Part D: Systematic Technique Benchmarking

Systematic comparison of **all techniques** on the SAME task and dataset.

**Metrics collected**:
- Training time (seconds)
- Peak memory usage (GB)
- Trainable parameter count & percentage
- Eval loss
- In-domain task accuracy / ROUGE / BERTScore
- Out-of-domain task accuracy (generalization)
- Inference speed (tokens/sec)
- Adapter size on disk (MB)
- Safety retention (refusal rate delta)

**Visualizations generated** (exported as SVG for HTML site):

1. **Radar chart** — multi-dimensional comparison of all techniques
2. **Pareto frontier** — memory vs quality trade-off
3. **Training curves overlay** — loss vs steps for all techniques
4. **Parameter efficiency bar chart** — trainable % for each method
5. **ID vs OOD scatter** — in-domain gain vs out-of-domain degradation per technique
6. **Decision flowchart** — "Which technique should I use?" logic tree

**Summary recommendation matrix**:

| Scenario | Recommended Technique | Why |
|---|---|---|
| Lots of data, no memory constraints | Full Fine-tuning | Best quality |
| Limited memory, good quality needed | LoRA | Best balance |
| LoRA quality plateau, need more | **DoRA** | Closes gap to full FT at LoRA cost |
| Very limited memory, large model | QLoRA | Enables the impossible |
| Minimal compute, task-specific | Prompt Tuning | Smallest footprint |
| Alignment / safety | DPO | No reward model needed |
| Multi-task from existing adapters | Model Merging | Zero additional training |
| Deploying tiny model | Knowledge Distillation | Compress capability |
| Best OOD generalization | DoRA or LoRA (low rank) | Less specialization tax |

---

## Phase 9 — Fun Use Cases

> **Steps 17–19** · Depends on Phases 3 & 5

### Step 17: Shakespearean Insult Generator (`14_fun_shakespearean_insult_generator.ipynb`)

| Attribute | Detail |
|---|---|
| **Technique** | LoRA on SmolLM2-360M |
| **Dataset** | Curated Shakespeare insults + plays excerpts from HF Hub |
| **What it does** | Generates creative Shakespearean insults on any topic |
| **Teaching point** | How style transfer works via fine-tuning |

- Show base model response vs fine-tuned response (side-by-side)
- Interactive demo cell: user types a topic → model generates an insult in iambic pentameter
- Evaluate: style adherence score, creativity metrics

**Example output**:
```
Input:  "Generate a Shakespearean insult about someone who writes bad code"
Output: "Thou artless, beetle-headed miscreant! Thy code doth read
         as if a drunken monkey, seized by fits, did pound upon
         the keys with reckless wild abandon, producing naught
         but tangled spaghetti fit for no compiler's gullet!"
```

### Step 18: Code Roast Reviewer (`15_fun_code_roast_reviewer.ipynb`)

| Attribute | Detail |
|---|---|
| **Technique** | QLoRA on Qwen2.5-1.5B-Instruct |
| **Dataset** | Synthetic — sarcastic-but-helpful code reviews generated from larger model |
| **What it does** | Gordon-Ramsay-style code reviews — brutal honesty with actionable advice |
| **Teaching point** | Controlling tone via fine-tuning while preserving technical accuracy; synthetic data generation pipeline |

- Build the synthetic data generation pipeline (use base Qwen2.5-3B to generate training data)
- Fine-tune with QLoRA (demonstrates memory savings for 1.5B model)
- Interactive demo: paste code → get roasted with actual improvement suggestions
- Evaluate: technical accuracy of suggestions + humor/tone rating

**Example output**:
```
Input:  [poorly written Python function with nested loops]
Output: "Oh BRILLIANT. You've managed to write an O(n³) solution to a problem
         that literally has an O(n) answer on the FIRST page of any algorithms
         textbook. This for-loop inception would make Christopher Nolan weep.

         Here's what you SHOULD do: Use a dictionary for O(1) lookups instead
         of this archaeological excavation through nested arrays..."
```

### Step 19: Dungeon Master (`16_fun_dungeon_master.ipynb`)

| Attribute | Detail |
|---|---|
| **Technique** | DPO on SmolLM2-360M |
| **Dataset** | D&D adventure transcripts + synthetic preference pairs (engaging vs boring narration) |
| **What it does** | Interactive text adventure — model narrates, user makes choices |
| **Teaching point** | How DPO shapes creative generation toward engaging, vivid storytelling |

- Create preference pairs: vivid/engaging narration (chosen) vs flat/boring narration (rejected)
- Two-stage: SFT for adventure narration → DPO for engagement quality
- Build a simple game loop in the notebook with state tracking
- Interactive play session: demonstrate 3–4 turns of gameplay

---

## Phase 10 — Novel Industry Use Cases

> **Steps 20–21** · Depends on Phases 3, 5, & 6 · These are the differentiators

### Step 20: Agentic Tool-Calling SLM (`17_industry_agentic_tool_calling.ipynb`)

> **Novel angle**: Fine-tune a SMALL model (360M parameters) to make reliable structured JSON function/tool calls — replacing expensive large model API calls for agentic workflows.

**Why this is novel**: Every existing tutorial on tool-calling uses GPT-4, Claude, or 70B+ models. Nobody has demonstrated that a 360M model can be a **reliable** tool-caller. This changes the economics of agentic AI.

**Safety connection** (ties to Phase 6): Tool-calling models must NOT hallucinate tool calls or fabricate parameter values — an incorrect `delete_database(table="production")` call is catastrophic. The notebook includes:
- Refusal behavior: model should say "I cannot determine..." for ambiguous queries rather than guessing
- Confidence calibration: structured uncertainty output
- Sandbox testing protocol before any real-tool connection

#### Implementation Plan

1. **Define tool schemas** (5–8 tools):
   - `get_weather(city, date)` → weather data
   - `query_database(table, filters, columns)` → SQL-like queries
   - `create_calendar_event(title, datetime, duration)` → scheduling
   - `calculate(expression)` → math evaluation
   - `search_knowledge_base(query, top_k)` → RAG retrieval
   - `send_notification(recipient, message, priority)` → alerting
   - `get_stock_price(ticker, period)` → financial data
   - `translate_text(text, source_lang, target_lang)` → translation

2. **Synthetic dataset generation**:
   - Generate 5,000+ user queries → structured JSON tool-call pairs
   - Include: single tool calls, multi-tool chains, ambiguous queries (should ask for clarification), **adversarial inputs that should be refused**
   - Format: `{"user": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}`

3. **Fine-tuning**: LoRA on SmolLM2-360M with structured output training

4. **Evaluation harness**:
   - JSON validity rate (syntactically correct JSON)
   - Schema compliance rate (correct tool name + parameter names)
   - Parameter accuracy (correct values extracted from query)
   - Multi-tool routing accuracy (picks right tool for ambiguous queries)
   - **Safety: refusal rate on adversarial/ambiguous inputs** (should be high)

5. **Results showcase**:
   ```
   Metric              | Base Model | Fine-tuned | Improvement
   --------------------|------------|------------|------------
   JSON validity       | ~30%       | ~96%       | +66%
   Schema compliance   | ~15%       | ~93%       | +78%
   Parameter accuracy  | ~10%       | ~89%       | +79%
   Appropriate refusal | ~5%        | ~82%       | +77%
   ```

**Industry value**: Edge deployment, cost reduction (100x cheaper than GPT-4 API), latency-critical agentic systems, offline-capable agents.

### Step 21: Privacy-Preserving Clinical Note Summarizer (`18_industry_privacy_preserving_clinical_summarizer.ipynb`)

> **Novel angle**: Two-stage fine-tuning pipeline (QLoRA → DPO) for clinical summarization that runs ENTIRELY on-device — zero patient data leaves the machine. Privacy-by-architecture for HIPAA/GDPR compliance.

**Why this is novel**: Combines three things no existing tutorial covers together:
1. Synthetic clinical data generation from public medical ontologies
2. Two-stage fine-tuning (capability + alignment for factual accuracy)
3. On-device inference as a privacy architecture decision

**Safety connection** (ties to Phase 6): Clinical summarization has life-or-death stakes. Hallucinated medications or fabricated lab values could harm patients. The DPO stage specifically targets:
- Factual grounding (every claim traceable to source note)
- Uncertainty expression ("The note does not clearly indicate..." rather than fabricating)
- Scope limitation (model refuses to provide medical advice, only summarizes)

#### Implementation Plan

1. **Synthetic data generation pipeline**:
   - Use public medical ontologies (SNOMED CT concepts, ICD-10 codes)
   - Generate synthetic clinical encounters with known ground-truth summaries
   - Create 2,000+ synthetic notes covering: chief complaint, history, exam findings, assessment, plan
   - NO real patient data at any point
   - **Data governance**: Document all sources, licensing (SNOMED CT license terms), and synthetic generation methodology

2. **Stage 1 — QLoRA for summarization capability**:
   - Fine-tune Qwen2.5-1.5B on clinical note → structured summary task
   - Output format: Chief Complaint, Key Findings, Assessment, Plan (structured sections)

3. **Stage 2 — DPO for clinical accuracy**:
   - Create preference pairs: factually accurate summary (chosen) vs hallucinated summary (rejected)
   - Hallucination types: made-up medications, wrong dosages, fabricated lab values, invented diagnoses
   - DPO to reduce hallucination rate and improve factual grounding

4. **On-device inference demo**:
   - Run entirely on MPS (M4 Max) — simulates edge hospital workstation
   - Measure: inference latency, memory footprint
   - Show: no network calls during inference (verify with network monitor)

5. **Evaluation**:
   - Factual accuracy: % of summary facts traceable to source note
   - Hallucination rate: % of fabricated information in summaries
   - Summary completeness: % of critical information captured
   - Before/after DPO comparison
   - **OOD test**: Evaluate on note types not seen during training

**Industry value**: Healthcare (HIPAA), finance (SOX), legal (privilege), government (classified) — any domain with strict data residency requirements.

---

## Phase 11 — Apple Silicon Optimization

> **Step 22** · Can run parallel with other phases; finalize after all notebooks

### Step 22: Apple Silicon Optimization Guide (`19_apple_silicon_optimization_guide.ipynb`)

- **MPS backend deep-dive**: what operations are supported, what falls back to CPU, common workarounds
- **Unified memory advantage**: demonstrate loading a model that wouldn't fit on an 8GB GPU
- **Memory monitoring**: real-time MPS memory tracking utilities
- **Performance tuning experiments**:

  | Setting | Impact | Recommendation |
  |---|---|---|
  | Batch size | Linear memory scaling | Start with 2, increase until OOM |
  | Gradient accumulation | Memory-neutral throughput increase | Use 4–8 steps |
  | Mixed precision (fp16/bf16) | ~50% memory reduction | Always enable on M4 |
  | Gradient checkpointing | ~30% memory reduction, ~20% slower | Enable for >1B models |

- **MLX comparison**: Same LoRA fine-tuning task implemented in MLX framework
- **Benchmark table**: M4 Max vs cloud GPU (A100/H100 reference numbers from papers)
- **Production deployment tips**: CoreML export, GGUF conversion, on-device serving

---

## Phase 12 — Interactive GitHub Pages Site

> **Steps 23–26** · Structure can start early; content depends on notebook outputs

### Step 23: Site Infrastructure

- **Framework**: Tailwind CSS via CDN (no build step), vanilla JavaScript
- **Features**: Responsive design, dark/light mode toggle, smooth navigation
- **Deployment**: GitHub Actions workflow (`.github/workflows/pages.yml`)
- **Layout**: Consistent header/footer, sidebar navigation on technique pages

### Step 24: Technique & Decision Pages (`techniques.html`, `comparison.html`, `choosing-method.html`)

**`techniques.html`** — Per-technique sections:
- How it works (SVG architecture diagram)
- Mathematical formulation (rendered with KaTeX)
- When to use / when NOT to use (bullet points)
- Pros/cons table
- Code snippet preview (syntax highlighted)
- Direct link to corresponding notebook

**`comparison.html`** — Interactive elements:
- Sortable comparison table (sort by memory, speed, quality, parameter count)
- Interactive Pareto chart (hover for details)
- Decision flowchart with clickable nodes
- Filter: "I have X GB memory" → highlights viable techniques

**`choosing-method.html`** — Task-to-technique decision guide:
- Interactive flowchart (from the "Choosing a Fine-Tuning Method" section)
- Task cards: click a task objective → see recommended technique + rationale
- "Method Recommender": answer 3–4 questions → get personalized recommendation

### Step 25: Safety, Failure & Use Case Pages

**`safety.html`** — Responsible fine-tuning guide:
- Risk taxonomy visual (SVG)
- Before/after safety metrics dashboard mockup
- Mitigation strategy cards
- Governance checklist (interactive, checkable)

**`failure-points.html`**:
- Visual catalog with severity badges (🔴🟠🟡)
- Each failure: symptom → cause → fix (collapsible accordion)
- "Quiz" mode: see a training curve, identify the problem

**`fun-usecases.html`**:
- Showcase cards with sample outputs
- Technique badge, difficulty rating, time estimate
- Key learnings highlighted

**`industry-usecases.html`**:
- Architecture diagrams (SVG)
- Business value proposition
- Safety considerations highlighted
- Deployment considerations
- Results tables and charts

### Step 26: Landing Page & Learning Path (`index.html`, `guide.html`)

**`index.html`**:
- Hero section with project tagline and key stats (10+ techniques, 20 notebooks, 5 use cases)
- Visual learning path (branching for different experience levels)
- Quick-start guide (3 steps to first fine-tune)
- Links to all pages

**`guide.html`**:
- Step-by-step curriculum with estimated time per section
- Three tracks: Beginner (5 notebooks), Intermediate (10 notebooks), Advanced (all 20)
- Interactive progress tracker (localStorage-based)

---

## Phase 13 — Polish & Academic Readiness

> **Steps 27–28** · Depends on ALL prior phases

### Step 27: Documentation & Packaging

- Comprehensive `README.md` with: badges, table of contents, quickstart, screenshots
- Citation information (BibTeX entry for academic use)
- Contributing guidelines (`CONTRIBUTING.md`)
- License: MIT
- `.gitignore` for model checkpoints, data cache, outputs
- Data governance documentation (`data/README.md`) with licensing and sourcing details

### Step 28: Final Validation

- [ ] Run all 20 notebooks end-to-end on M4 Max
- [ ] Verify no notebook exceeds 48GB peak memory
- [ ] All failure-point cells produce the failure + fix
- [ ] Safety notebook demonstrates measurable guardrail erosion + recovery
- [ ] Evaluation notebook shows clear ID vs OOD divergence
- [ ] All visualizations are publication-ready (labels, legends, fonts)
- [ ] HTML site renders correctly in Chrome, Safari, Firefox
- [ ] GitHub Pages deploys successfully
- [ ] All links work (notebooks ↔ HTML pages)
- [ ] Proofread all explanations for technical accuracy
- [ ] Test on fresh clone: can a new user follow the quickstart?

---

## Dependency Graph

```
Phase 1 (Foundation + Dataset Design)
  │
  ├──► Phase 2 (Basic Techniques)
  │       │
  ├──► Phase 3 (PEFT: LoRA, DoRA, QLoRA) ◄────────┐
  │       │                                         │
  ├──► Phase 4 (Prompt-Based) ║ parallel            │
  │       │                                         │
  │       ▼                                         │
  ├──► Phase 5 (Advanced: DPO, Distill, Merge) ───►│
  │       │                                         │
  │       ▼                                         │
  ├──► Phase 6 (Safety & Responsible FT) ◄─────────┘
  │       │
  ├──► Phase 7 (Failure Analysis) ◄── needs Phases 2–5
  │       │
  ├──► Phase 8 (Evaluation Theory + Benchmarking) ◄── needs all techniques
  │       │
  ├──► Phase 9 (Fun Use Cases) ◄── needs LoRA, QLoRA, DPO
  │       │
  ├──► Phase 10 (Industry Use Cases) ◄── needs LoRA, QLoRA, DPO + Safety
  │       │
  ├──► Phase 11 (Apple Silicon) ║ parallel, finalize last
  │       │
  ├──► Phase 12 (HTML Site) ◄── content from all phases
  │       │
  └──► Phase 13 (Polish) ◄── depends on EVERYTHING
```

---

## Verification Checklist

| Check | Criteria | Phase |
|---|---|---|
| Environment | All imports succeed, MPS detected, inference works | 1 |
| Notebook runtime | Each notebook < 30 min on M4 Max | All |
| Memory safety | No notebook > 48 GB peak unified memory | All |
| Quality | Each fine-tuned model measurably improves over base | 2–10 |
| DoRA vs LoRA | DoRA shows measurable quality improvement over LoRA | 3 |
| Safety | Guardrail erosion measured + mitigated in demo | 6 |
| Evaluation | ID vs OOD divergence clearly demonstrated | 8 |
| Dataset design | Notebook covers governance, licensing, labeling | 1 |
| Failure demos | Each failure cell produces the failure, then the fix | 7 |
| Charts | All charts have titles, axis labels, legends, proper fonts | 8 |
| HTML site | All pages render; images load; interactions work | 12 |
| GitHub Pages | Deploys at `kumarjit-pathak.github.io/finetuning-slm` | 12 |
| Academic quality | Technically accurate, citable, presentable | 13 |
| Reproducibility | Fresh clone + quickstart works for a new user | 13 |

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model family | SmolLM2 + Qwen2.5 | Well-supported, great size range (135M–3B), active community |
| Include DoRA | Yes, in flagship LoRA notebook | Emerging best practice; supported in PEFT; bridges LoRA↔Full FT gap |
| Safety as dedicated phase | Yes (Phase 6) | Critical for academic credibility and industry use cases |
| Evaluation theory | Expanded with ID/OOD + metric limitations | Addresses "metrics can mislead" gap; essential for researchers |
| Dataset design | Expanded in Phase 1 | Foundation that informs all subsequent fine-tuning decisions |
| Python environment | Existing `agentic-slm` | Already has all core packages; avoid environment proliferation |
| HTML framework | Tailwind CSS via CDN | No build step, beautiful defaults, GitHub Pages compatible |
| Interactivity | Vanilla JS | No framework overhead for a static educational site |
| Infographics | SVG (programmatic + hand-crafted) | Scalable, editable, version-controllable |
| Datasets | HF Hub + synthetic generation | No proprietary data; fully reproducible |
| Scope boundary | Fine-tuning only | NOT pre-training, NOT serving infrastructure (beyond inference demos) |

---

## Appendix A — Emerging Methods

> Acknowledges newer PEFT and alignment approaches the tutorial is aware of but not yet implementing in detail. Positions them as future extensions and provides readers with pointers for further exploration.

### Emerging PEFT Variants

| Method | Paper | Key Innovation | Status |
|---|---|---|---|
| **DoRA** | Liu et al., 2024 | Weight-Decomposed Low-Rank Adaptation — separates magnitude and direction | ✅ **Implemented** in this tutorial (Phase 3) |
| **GaLore** | Zhao et al., 2024 | Gradient Low-Rank Projection — projects gradients to low-rank space for memory-efficient full-rank training | Promising for full FT memory reduction; not yet in PEFT |
| **ReLoRA** | Lialin et al., 2023 | Stacked Low-Rank — periodically merge LoRA into base and restart with new LoRA | Allows gradual full-rank updates; experimental |
| **LoRA+** | Hayou et al., 2024 | Different learning rates for A and B matrices (B needs higher LR) | Simple improvement; easy to adopt |
| **AdaLoRA** | Zhang et al., 2023 | Adaptive rank allocation — dynamically allocates rank budget across layers based on importance scores | PEFT-supported; useful when unsure about rank distribution |
| **LoRA-FA** | Zhang et al., 2023 | Freezes A matrix after initialization, only trains B — halves trainable params | Ultra-low resource scenarios |
| **LongLoRA** | Chen et al., 2023 | Efficient fine-tuning for long-context extension using shifted sparse attention | Relevant for context length extension tasks |
| **VeRA** | Kopiczko et al., 2024 | Vector-based Random Matrix Adaptation — shared random matrices, per-layer scaling vectors | Extreme parameter efficiency; 10x fewer params than LoRA |
| **FourierFT** | Gao et al., 2024 | Learns spectral components in Fourier space rather than spatial weight updates | Novel approach; early results promising |

### Emerging Alignment Methods

| Method | Paper | Key Innovation | Relation to DPO |
|---|---|---|---|
| **ORPO** | Hong et al., 2024 | Odds Ratio Preference Optimization — combines SFT and alignment in single stage | Eliminates need for separate SFT step; simpler pipeline |
| **KTO** | Ethayarajh et al., 2024 | Kahneman-Tversky Optimization — works with binary (good/bad) labels, no paired preferences needed | Easier data collection; works when paired preferences unavailable |
| **SimPO** | Meng et al., 2024 | Simple Preference Optimization — reference-free, uses average log probability as implicit reward | Simpler than DPO; no reference model needed |
| **RLOO** | Ahmadian et al., 2024 | REINFORCE Leave-One-Out — online RL variant that outperforms DPO on some benchmarks | Bridges gap between DPO and full RL approaches |
| **IPO** | Azar et al., 2023 | Identity Preference Optimization — addresses DPO's overfitting to deterministic preferences | More robust to noisy preference data |
| **cDPO** | Mitchell et al., 2023 | Conservative DPO — adds KL constraint to prevent reward hacking | Addresses DPO failure mode (Phase 5, failure point #2) |

### When to Consider Emerging Methods

- **Already using LoRA and hitting quality ceiling** → Try DoRA (implemented here) or AdaLoRA
- **Need full-FT quality with LoRA memory** → Investigate GaLore or ReLoRA (experimental)
- **Can't collect paired preferences for DPO** → KTO works with unpaired good/bad labels
- **Want simpler alignment pipeline** → ORPO eliminates separate SFT stage
- **DPO is overfitting to preferences** → Try IPO or cDPO for robustness

### Why These Are "Appendix" Not "Core"

1. **Maturity**: Most lack production-grade library support (exceptions: DoRA, AdaLoRA in PEFT)
2. **Reproducibility**: Limited community validation beyond original papers
3. **Incremental gains**: Most offer 2–5% improvements over well-tuned LoRA/DPO
4. **Complexity**: Added implementation complexity rarely justified for educational setting

> **Future plan**: As methods mature and enter PEFT/TRL mainline, promote the most impactful ones from this appendix into dedicated tutorial sections.

---

## Future Enhancements (Out of Scope for v1)

- [ ] **SmolLM3 comparison** — add benchmarks once model stabilizes
- [ ] **Gradio Space** — deploy Code Roaster as live HuggingFace Space
- [ ] **MLX-native track** — full fine-tuning pipeline using Apple's MLX framework
- [ ] **Video tutorials** — narrated walkthrough of each notebook
- [ ] **RLHF notebook** — full reinforcement learning from human feedback (requires reward model training)
- [ ] **Multi-modal fine-tuning** — extend to vision-language SLMs (SmolVLM)
- [ ] **GaLore implementation** — once library support stabilizes
- [ ] **ORPO notebook** — single-stage alignment alternative to SFT+DPO pipeline
- [ ] **Federated fine-tuning** — distributed privacy-preserving fine-tuning across institutions
