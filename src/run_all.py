"""
Runs every PEFT method for a given base model and label column, saving results incrementally
to a CSV so a crashed/interrupted run can be resumed without repeating completed methods.

Usage (from the repo root):
    python src/run_all.py --model_name Qwen/Qwen2.5-1.5B --label_col label_domain
    python src/run_all.py --model_name Qwen/Qwen2.5-7B --label_col label_domain
    python src/run_all.py --model_name Qwen/Qwen2.5-1.5B --label_col label_fine
"""

import argparse
import os
import torch
import pandas as pd

from train import build_and_train
from methods import get_method_configs


def run_all(model_name: str, label_col: str, epochs: int, batch_size: int,
            results_dir: str = "results", methods: list | None = None) -> pd.DataFrame:
    os.makedirs(results_dir, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    results_fp = os.path.join(results_dir, f"results_{safe_model_name}_{label_col}.csv")

    if methods is None:
        methods = list(get_method_configs().keys())

    # Resume support: skip methods already completed in a previous run
    if os.path.exists(results_fp):
        existing = pd.read_csv(results_fp)
        done = set(existing["method"].tolist())
        print(f"Found existing results for: {sorted(done)} — will skip these.")
    else:
        existing = pd.DataFrame()
        done = set()

    all_results = existing.to_dict("records")

    for method_name in methods:
        if method_name in done:
            continue
        print(f"\n{'='*50}\nTraining: {method_name}\n{'='*50}")
        try:
            result = build_and_train(
                method_name=method_name,
                model_name=model_name,
                label_col=label_col,
                epochs=epochs,
                batch_size=batch_size,
            )
            all_results.append(result)
        except Exception as e:
            print(f"FAILED on {method_name}: {e}")
            all_results.append({"method": method_name, "error": str(e)})

        # Save after every method — this is what makes the run resumable
        pd.DataFrame(all_results).to_csv(results_fp, index=False)
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    print(f"\nAll done. Results saved to: {results_fp}")
    print(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--label_col", default="label_domain", choices=["label_domain", "label_fine"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=None,
                         help="Subset of methods to run, e.g. --methods lora qlora")
    args = parser.parse_args()

    run_all(
        model_name=args.model_name,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        methods=args.methods,
    )
