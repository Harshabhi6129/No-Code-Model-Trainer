# backend/dataset_validator.py
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from llm_service import query_llm_gemini


def validate_csv(file_path: str) -> Dict:
    """Validate and analyze a CSV file."""
    df = pd.read_csv(file_path)
    filename = Path(file_path).name
    return analyze_dataset(df, filename)


def _label_plot(df: pd.DataFrame) -> Optional[str]:
    if "label" not in df.columns:
        return None
    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df, palette="pastel")
    plt.title("Label distribution")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _numeric_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in df.select_dtypes(include="number").columns:
        stats[col] = {
            "mean": float(df[col].mean(skipna=True)),
            "median": float(df[col].median(skipna=True)),
            "std": float(df[col].std(skipna=True)),
            "min": float(df[col].min(skipna=True)),
            "max": float(df[col].max(skipna=True)),
        }
    return stats


def analyze_dataset(df: pd.DataFrame, filename: str) -> Dict:
    """Validate the dataframe and return rich metadata used by the UI."""
    required = {"text", "label"}
    missing_req = required - set(df.columns)
    warnings: List[str] = []
    suggestions: List[str] = []

    valid = not missing_req
    if missing_req:
        warnings.append(f"Missing columns: {', '.join(missing_req)}")
        suggestions.append("CSV must contain 'text' and 'label' columns.")

    # class-imbalance notice
    if "label" in df.columns:
        freq = df["label"].value_counts(normalize=True).to_dict()
        for lbl, pct in freq.items():
            if pct > 0.75:
                warnings.append(
                    f"Class '{lbl}' dominates ({pct * 100:.1f}%). Consider balancing."
                )

    # column quick-summary
    col_meta: List[Dict] = []
    for col in df.columns:
        col_meta.append(
            {
                "name": col,
                "type": str(df[col].dtype),
                "unique": int(df[col].nunique()),
                "sample": df[col].dropna().astype(str).unique()[:3].tolist(),
            }
        )

    # LLM insights (short paragraph, no markdown)
    try:
        llm_insights = query_llm_gemini(
            f"""Give a short paragraph of insights about this dataset:
Filename: {filename}
Columns: {list(df.columns)}
Head:
{df.head(5).to_string(index=False)}
"""
        )
    except Exception as err:
        llm_insights = f"LLM error: {err}"

    # lightweight model catalogue (transformer focus only)
    model_catalog = {
        "transformers": [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base",
            "albert-base-v2",
            "google/electra-small-discriminator",
        ],
        "advice": "Start with DistilBERT for small datasets, or RoBERTa for higher accuracy.",
    }

    return {
        "valid": valid,
        "warnings": warnings,
        "suggestions": suggestions,
        "preview": df.head(5).to_dict(orient="records"),
        "columns": col_meta,
        "numeric_stats": _numeric_stats(df),
        "missing_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "plot_base64": _label_plot(df),
        "llm_insights": llm_insights,
        "server_path": f"uploads/{filename}",
        "model_catalog": model_catalog,
    }
