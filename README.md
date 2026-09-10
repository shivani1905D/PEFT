# PEFT + Quantization Comparative Benchmark

Compares 6 parameter-efficient fine-tuning methods — LoRA, QLoRA, Prefix-Tuning, Prompt-Tuning,
P-Tuning, LoftQ — on Web of Science (WOS-46985) scientific abstract classification, measuring
accuracy, F1, training time, peak GPU memory, and trainable parameter count for each.

> **Note on QA-LoRA / QDLoRA:** these two do not have official `peft` library support (only the
> original authors' research code). They are excluded from the automated pipeline here; see
> "Extending to QA-LoRA / QDLoRA" below if you want to add them manually.

## Repo structure

```
├── requirements.txt
├── src/
│   ├── data_utils.py   # downloads WOS-46985, builds the fixed train/val/test split
│   ├── methods.py      # the 6 PEFT/quantization method configs
│   ├── train.py         # trains ONE method, CLI: python src/train.py --method lora
│   └── run_all.py       # trains ALL methods for a model, resumable, CLI
├── results/              # output CSVs land here (one per model+label combination)
└── notebooks/
    └── colab_runner.ipynb  # minimal notebook — just clones + calls the scripts above
```

## Why scripts instead of notebook cells

Every error you'll hit repeatedly pasting code into live Colab cells (stale imports after a
`pip install`, mismatched library versions across cells) goes away when you run `python script.py`
instead — each invocation is a **fresh process**, so there's no leftover state from a previous
install or import to conflict with.

## Setup (run once per fresh Colab runtime)

```python
!git clone https://github.com/YOUR_USERNAME/peft-quant-benchmark.git
%cd peft-quant-benchmark
!pip install -q -r requirements.txt
```

**Then: `Runtime` → `Restart session`.** Do this every time you change `requirements.txt` or
reinstall anything — never skip it, it's what prevents the version-mismatch crashes.

After restarting:
```python
%cd peft-quant-benchmark
```
(you only need to `cd` back in — no need to reinstall or re-clone)

## Running

**Sanity-check one method first** (recommended before committing a session to the full loop):
```python
!python src/train.py --method lora --model_name Qwen/Qwen2.5-1.5B --label_col label_domain
```

**Run all 6 methods for one model:**
```python
!python src/run_all.py --model_name Qwen/Qwen2.5-1.5B --label_col label_domain
```

**Run the larger model** (loads in 4-bit automatically for the non-quantized methods too, to fit
a T4's memory — see `methods.py` if you want to change this):
```python
!python src/run_all.py --model_name Qwen/Qwen2.5-7B --label_col label_domain
```

**Hard-mode run on the 134-class fine-grained labels**, once the 7-class runs look good:
```python
!python src/run_all.py --model_name Qwen/Qwen2.5-1.5B --label_col label_fine
```

**Run only specific methods** (useful for retrying a single failed one):
```python
!python src/run_all.py --model_name Qwen/Qwen2.5-1.5B --label_col label_domain --methods qlora loftq
```

## Resuming an interrupted run

`run_all.py` saves results to `results/results_<model>_<label_col>.csv` **after every single
method**, and automatically skips any method already present in that file on the next run. If
Colab disconnects mid-way, just re-run the exact same command — it picks up where it left off.

## Getting results out to GitHub

```python
!git add results/
!git commit -m "Add results for Qwen2.5-1.5B, label_domain"
!git push
```
(You'll need a GitHub Personal Access Token the first time you push from Colab — see GitHub's
docs on using a token in place of a password for HTTPS git operations.)

## Reading the results

Each row of the output CSV is one method, with columns:
`method, model_name, label_col, num_labels, trainable_params, total_params, trainable_pct,
train_time_sec, peak_mem_gb, test_accuracy, test_f1_macro` — directly matching the four metrics
defined in the project's review paper (Table IV).

## Extending to QA-LoRA / QDLoRA

Both require quantization-scheme code not available in `peft`. To add them:
1. Clone the original authors' repos (QA-LoRA: search their GitHub; QDLoRA: same).
2. Add a new file `src/methods_extra.py` implementing their custom quantize+adapter setup.
3. Wire a new `--method qa_lora` / `--method qdlora` branch into `train.py`'s model-loading step.

This is meaningfully more engineering work than the 6 methods above, which is exactly the
"narrow, pairwise comparison" gap the review paper's Section IV-E discusses — treat this as a
stretch goal after the core 6-method comparison is complete.
