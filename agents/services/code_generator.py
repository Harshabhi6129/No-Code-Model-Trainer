"""
Standalone training script generator.

Takes pipeline artifacts (TaskSpec + DataProfile + ModelRecipe + TrainingResult)
and emits a self-contained, copy-paste-runnable Python script.

The script is AST-validated before returning to guarantee syntax correctness.
"""
from __future__ import annotations

import ast
import json
from typing import Any


class CodeGenerationError(Exception):
    """Raised when the generated script fails AST validation."""


def generate_training_script(
    task_spec:       dict[str, Any],
    data_profile:    dict[str, Any],
    model_recipe:    dict[str, Any],
    training_result: dict[str, Any],
) -> str:
    """
    Generate a standalone Python training script from pipeline artifacts.

    Returns:
        str — the script source code.

    Raises:
        CodeGenerationError — if the output fails ast.parse().
    """
    approach   = str(model_recipe.get("training_approach", "full_finetune"))
    base_model = str(model_recipe.get("base_model", "distilbert-base-uncased"))

    # Resolve label names (prefer training_result → task_spec → data_profile)
    label_names = (
        training_result.get("label_names")
        or task_spec.get("label_names")
        or list(data_profile.get("label_distribution", {}).keys())
        or []
    )
    label_names = [str(l) for l in label_names]

    is_lora  = approach in ("lora", "qlora")
    is_qlora = approach == "qlora"

    lr       = float(model_recipe.get("learning_rate", 2e-5) or 2e-5)
    epochs   = int(model_recipe.get("num_epochs", 3) or 3)
    batch    = int(model_recipe.get("batch_size", 16) or 16)
    max_len  = int(model_recipe.get("max_length", 128) or 128)
    warmup   = float(model_recipe.get("warmup_ratio", 0.1) or 0.1)
    wd       = float(model_recipe.get("weight_decay", 0.01) or 0.01)
    lora_r   = int(model_recipe.get("lora_r", 16) or 16)
    lora_alpha = int(model_recipe.get("lora_alpha") or lora_r * 2)
    input_col  = str(task_spec.get("input_column", "text"))
    label_col  = str(task_spec.get("label_column", "label"))

    acc = training_result.get("accuracy")
    f1  = training_result.get("f1")
    metrics_str = (
        f"Accuracy={acc*100:.1f}%  F1={f1:.3f}"
        if acc is not None and f1 is not None
        else "metrics unavailable"
    )

    # ── Build sections ─────────────────────────────────────────────────────────

    extra_imports = []
    if is_qlora:
        extra_imports.append("from transformers import BitsAndBytesConfig")

    extra_imports_str = "\n".join(extra_imports)

    # CONFIG dict lines
    config_lines = [
        f'    "base_model":         {json.dumps(base_model)},',
        f'    "training_approach":  {json.dumps(approach)},',
        f'    "label_names":        {json.dumps(label_names)},',
        f'    "input_column":       {json.dumps(input_col)},',
        f'    "label_column":       {json.dumps(label_col)},',
        f'    "learning_rate":      {lr!r},',
        f'    "num_epochs":         {epochs},',
        f'    "batch_size":         {batch},',
        f'    "max_length":         {max_len},',
        f'    "warmup_ratio":       {warmup},',
        f'    "weight_decay":       {wd},',
    ]
    config_block = "\n".join(config_lines)

    # QLoRA config block (inside train())
    if is_qlora:
        bnb_block = (
            "    bnb_config = BitsAndBytesConfig(\n"
            "        load_in_4bit=True,\n"
            '        bnb_4bit_quant_type="nf4",\n'
            "        bnb_4bit_use_double_quant=True,\n"
            "        bnb_4bit_compute_dtype=torch.bfloat16,\n"
            "    )"
        )
        model_load_block = (
            '    model = AutoModelForSequenceClassification.from_pretrained(\n'
            '        CONFIG["base_model"],\n'
            '        num_labels=len(CONFIG["label_names"]),\n'
            '        quantization_config=bnb_config,\n'
            '        device_map="auto",\n'
            '    )'
        )
    else:
        bnb_block = ""
        model_load_block = (
            '    model = AutoModelForSequenceClassification.from_pretrained(\n'
            '        CONFIG["base_model"],\n'
            '        num_labels=len(CONFIG["label_names"]),\n'
            '    )'
        )

    # LoRA adapter block (inside train(), after model load)
    if is_lora:
        lora_block = (
            "    # ── LoRA adapter ─────────────────────────────────────────────────\n"
            "    from peft import get_peft_model, LoraConfig, TaskType\n"
            "    lora_cfg = LoraConfig(\n"
            "        task_type=TaskType.SEQ_CLS,\n"
            f"        r={lora_r},\n"
            f"        lora_alpha={lora_alpha},\n"
            '        target_modules=["query", "key", "value"],\n'
            '        bias="none",\n'
            "    )\n"
            "    model = get_peft_model(model, lora_cfg)\n"
            "    model.print_trainable_parameters()"
        )
    else:
        lora_block = ""

    # Build the full script using explicit string concatenation
    sections = []

    sections.append(f'''\
#!/usr/bin/env python3
"""
Auto-generated training script by ModelForge.
Run result: {metrics_str}
Base model: {base_model}  Approach: {approach}

Usage:
    pip install transformers datasets peft accelerate scikit-learn
    python train.py --data_path path/to/your/data.csv
"""
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)''')

    if extra_imports_str:
        sections.append(extra_imports_str)

    sections.append(f'''\

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {{
{config_block}
}}''')

    sections.append('''\

def load_data(data_path: str):
    df = pd.read_csv(data_path)
    assert CONFIG["input_column"] in df.columns, f"Missing column: {CONFIG['input_column']}"
    assert CONFIG["label_column"] in df.columns, f"Missing column: {CONFIG['label_column']}"
    df = df.dropna(subset=[CONFIG["input_column"], CONFIG["label_column"]])
    df[CONFIG["input_column"]] = df[CONFIG["input_column"]].astype(str)
    df[CONFIG["label_column"]] = df[CONFIG["label_column"]].astype(str)

    label_names = CONFIG["label_names"] or sorted(df[CONFIG["label_column"]].unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(label_names)}
    df["label_int"] = df[CONFIG["label_column"]].map(label2id)
    df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df["label_int"] if df["label_int"].nunique() > 1 else None,
    )
    return train_df, test_df, label_names


def tokenize(df, tokenizer):
    hf_ds = Dataset.from_dict({
        "text":  df[CONFIG["input_column"]].tolist(),
        "label": df["label_int"].tolist(),
    })
    return hf_ds.map(
        lambda b: tokenizer(
            b["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=False,
        ),
        batched=True,
    )


def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    refs  = p.label_ids
    return {
        "accuracy": accuracy_score(refs, preds),
        "f1":       f1_score(refs, preds, average="weighted", zero_division=0),
    }


def train(data_path: str, output_dir: str = "trained_model"):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    train_df, test_df, label_names = load_data(data_path)
    train_ds = tokenize(train_df, tokenizer)
    test_ds  = tokenize(test_df,  tokenizer)
''')

    if bnb_block:
        sections.append(bnb_block)

    sections.append(model_load_block)

    if lora_block:
        sections.append("")
        sections.append(lora_block)

    sections.append('''\

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"] * 2,
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\\nModel saved to {output_dir}")
    results = trainer.evaluate()
    print(f"Eval accuracy: {results.get('eval_accuracy', 0)*100:.1f}%")
    print(f"Eval F1:       {results.get('eval_f1', 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to your CSV dataset")
    parser.add_argument("--output_dir", default="trained_model")
    args = parser.parse_args()
    train(data_path=args.data_path, output_dir=args.output_dir)
''')

    script = "\n".join(sections)

    # Validate syntax before returning
    try:
        ast.parse(script)
    except SyntaxError as exc:
        raise CodeGenerationError(
            f"Generated script has a syntax error at line {exc.lineno}: {exc.msg}"
        ) from exc

    return script
