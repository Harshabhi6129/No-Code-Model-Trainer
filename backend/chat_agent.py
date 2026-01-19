"""
Lightweight chat-command handler.

Receives a run-id and free-text command, then returns a list of
{event: "...", ...payload} dictionaries that websocket_server will
immediately publish to the frontend.
"""
from __future__ import annotations
import json, base64
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from llm_service import query_llm_gemini

RUNS_DIR = Path(__file__).parent / "runs"


def _summarise(run_dir: Path) -> str:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    epochs  = max(m["epoch"] for m in metrics)
    best    = min((m for m in metrics if "val_loss" in m),
                  key=lambda x: x["val_loss"], default={})
    text = (
        f"Training finished after {epochs} epochs.\n"
        f"• Best val-loss {best.get('val_loss', '--'):.4f}\n"
        f"• Best val-acc  {best.get('val_acc', 0):.3f}"
    )
    # paraphrase with LLM for nicer tone
    try:
        text = query_llm_gemini(
            f"Rephrase the following summary in 2 cheerful sentences:\n{text}"
        )
    except Exception:
        pass
    return text


def _confusion_png(run_dir: Path) -> str:
    # Generate confusion matrix on-the-fly if missing
    png_path = run_dir / "confusion.png"
    if not png_path.exists():
        # need predictions & labels
        preds_file = run_dir / "val_preds.json"
        if not preds_file.exists():                   # fallback
            raise FileNotFoundError("No predictions stored.")
        data = json.loads(preds_file.read_text())
        cm   = confusion_matrix(data["y_true"], data["y_pred"])
        labels = [str(l) for l in sorted(set(data["y_true"]))]
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.ylabel("True"); plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
    return base64.b64encode(png_path.read_bytes()).decode()


# ───────────────────────────────────────────────────────────────────
def handle_command(run_id: str, text: str) -> List[Dict]:
    run_dir = RUNS_DIR / run_id
    text_lc = text.lower().strip()

    if "summary" in text_lc or "progress" in text_lc:
        return [{"event": "chat", "text": _summarise(run_dir)}]

    if "confusion" in text_lc:
        try:
            b64 = _confusion_png(run_dir)
            return [{"event": "confusion", "img": b64}]
        except Exception as e:
            return [{"event": "chat", "text": f"⚠️ Could not build confusion matrix: {e}"}]

    # fallback LLM answer
    try:
        reply = query_llm_gemini(
            f"You are a helpful ML tutor. Answer this question briefly:\n{text}"
        )
    except Exception:
        reply = "Sorry, I can't answer that right now."

    return [{"event": "chat", "text": reply}]
