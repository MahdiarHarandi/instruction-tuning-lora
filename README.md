# Instruction Tuning via LoRA

> Parameter-efficient supervised fine-tuning of **TinyLlama-1.1B-Chat** to
> improve instruction following — training only **0.20%** of the model's
> parameters and measuring the gain on the **IFEval** benchmark.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-purple)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Small chat models often *understand* a request but ignore its explicit
constraints ("all caps", "exactly seven words", "return JSON"). This project
fine-tunes TinyLlama-1.1B-Chat with **LoRA** on an instruction-following dataset
and shows, both qualitatively and with a standard benchmark, that lightweight
adaptation meaningfully improves constraint adherence on a single consumer GPU.

**Key points**
- **LoRA / PEFT:** rank 8, α 16, dropout 0.05 on the attention projections
  (`q/k/v/o_proj`) → **2.25 M trainable / 1.10 B total = 0.2044%**.
- **SFT** on 2,000 examples from
  `allenai/tulu-3-sft-personas-instruction-following`, formatted with the model's
  chat template and label-masked padding.
- **Quantitative evaluation** with **IFEval** via EleutherAI's
  `lm-evaluation-harness`, base vs. fine-tuned.
- Adapter (**12.5 MB**) saved separately and also **merged** into a standalone
  model (~2.1 GB).

---

## Results

IFEval (50-sample subset, `lm-evaluation-harness`):

| Metric | Base | LoRA SFT | Change |
|---|---|---|---|
| `inst_level_strict_acc` | 0.092 | **0.263** | **≈ 2.8×** |
| `inst_level_loose_acc`  | 0.092 | **0.290** | ≈ 3.1× |
| `prompt_level_strict_acc` | 0.060 | **0.100** | ≈ 1.7× |
| `prompt_level_loose_acc`  | 0.060 | **0.100** | ≈ 1.7× |

Fine-tuning roughly **tripled instruction-level accuracy**. Qualitative tests
(escalating slogan prompts: all-caps → end-with-"soul" → exactly-7-words → JSON)
show the tuned model respecting formatting constraints the base model ignores.

> The 50-sample limit is a compute constraint, so absolute numbers carry
> variance — the comparison is meaningful as a *relative* base-vs-tuned delta.

---

## Training Configuration

| | |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Dataset | `allenai/tulu-3-sft-personas-instruction-following` (2,000 samples) |
| LoRA | r=8, α=16, dropout=0.05, target = q/k/v/o_proj |
| Trainable params | 2,252,800 / 1,102,301,184 (0.2044%) |
| Schedule | 1 epoch, batch 1 × grad-accum 16, lr 2e-4, cosine, warmup 0.03, fp16 |
| Run | 125 steps, final train loss ≈ 1.32, ~15 min on one GPU |

---

## Repository Contents

```
.
├── NLP_CA4_Q2_Harandi_810199596.ipynb     # main: LoRA SFT + IFEval evaluation
├── NLP_CA4_Q1_Harandi_810199596.ipynb     # related: BART vs GPT-2 for Text-to-SQL
└── Q2_adapter&merged/
    ├── tinyllama_lora_adapter/             # LoRA adapter (~12.5 MB) + tokenizer
    └── tinyllama_lora_merged/              # merged standalone model + tokenizer
```

> Note: `NLP_CA4_Q1` is a separate experiment (encoder–decoder vs. decoder-only
> for Text-to-SQL) included here for reference; the instruction-tuning project is
> `NLP_CA4_Q2`. Consider splitting it into its own repository.

---

## Getting Started

```bash
pip install transformers peft datasets accelerate
pip install lm-eval            # for IFEval
```

Then open `NLP_CA4_Q2_...ipynb` (GPU recommended) and run top to bottom:
data prep → LoRA config → SFT → save/merge → qualitative tests → IFEval.

Use the fine-tuned model by loading the base model and attaching the included
LoRA adapter (the lightweight, 12.5 MB result that ships with this repo):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base, "Q2_adapter&merged/tinyllama_lora_adapter")
tok = AutoTokenizer.from_pretrained("Q2_adapter&merged/tinyllama_lora_adapter")
```

> The repo ships the **LoRA adapter weights** and the merged model's
> config/tokenizer. The full merged weight file (~2.1 GB) is not committed —
> regenerate it with `model.merge_and_unload()` (see the notebook) if you need a
> standalone model.

---

## What's Inside the Notebook

- Tokenizer comparison across TinyLlama / Gemma-3 / Llama-3.1 on English and
  Persian text, with analysis of why tokenization differs.
- Chat-template handling and why it matters for instruction-tuned models.
- SFT data pipeline with label masking on padding tokens.
- Adapter-vs-merged size comparison.
- Base-vs-tuned evaluation: qualitative prompts + IFEval table.

---

## Limitations & Next Steps

- [ ] Run IFEval on the full set (not just 50 samples) for stable numbers.
- [ ] Sweep LoRA rank / target modules and dataset size.
- [ ] Add a short "answer only, no explanation" constraint to curb the tuned
      model's tendency to over-explain.

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Mahdiar Harandi** — harandimahdiar@gmail.com
[GitHub](https://github.com/MahdiarHarandi) · [LinkedIn](https://www.linkedin.com/in/mahdiarharandi)
