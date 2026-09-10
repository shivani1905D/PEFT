"""
Core training logic: loads a base model, wraps it with a chosen PEFT method, trains it,
and returns the four benchmark metrics (accuracy, F1, training time, peak memory, trainable params).

Usage (single method, from the repo root):
    python src/train.py --method lora --model_name Qwen/Qwen2.5-1.5B --label_col label_domain
"""

import argparse
import time
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model
from sklearn.metrics import accuracy_score, f1_score

from data_utils import get_splits
from methods import get_bnb_config, get_method_configs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def prepare_datasets(train_df, val_df, test_df, label_col, tokenizer, max_length=256):
    def to_ds(df):
        d = Dataset.from_pandas(df[["text", label_col]].rename(columns={label_col: "label"}))
        return d.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True)

    return to_ds(train_df), to_ds(val_df), to_ds(test_df)


def build_and_train(
    method_name: str,
    model_name: str,
    label_col: str = "label_domain",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    max_length: int = 256,
    output_dir: str = "runs",
) -> dict:
    train_df, val_df, test_df = get_splits()
    num_labels = train_df[label_col].nunique()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tok, val_tok, test_tok = prepare_datasets(
        train_df, val_df, test_df, label_col, tokenizer, max_length
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    cfg = get_method_configs()[method_name]
    load_kwargs = {"num_labels": num_labels}
    if cfg["quantize"]:
        load_kwargs["quantization_config"] = get_bnb_config()
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.bfloat16

    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(base_model, cfg["peft_config"])

    trainable_params, all_params = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=f"{output_dir}/{method_name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        bf16=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    test_metrics = trainer.evaluate(test_tok)

    return {
        "method": method_name,
        "model_name": model_name,
        "label_col": label_col,
        "num_labels": num_labels,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "trainable_pct": round(100 * trainable_params / all_params, 4),
        "train_time_sec": round(train_time, 1),
        "peak_mem_gb": round(peak_mem, 3),
        "test_accuracy": round(test_metrics["eval_accuracy"], 4),
        "test_f1_macro": round(test_metrics["eval_f1_macro"], 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(get_method_configs().keys()))
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--label_col", default="label_domain", choices=["label_domain", "label_fine"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    result = build_and_train(
        method_name=args.method,
        model_name=args.model_name,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("\n=== RESULT ===")
    print(result)
