"""
Converts a saved HuggingFace model to ONNX or TorchScript.
Called via asyncio.to_thread() — this blocks for 30-120 seconds.
"""
import logging
import tempfile
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def export_model(
    artifact_path: str,
    export_format: Literal["onnx", "torchscript"],
    opset_version: int = 14,
    optimize: bool = True,
) -> Path:
    """
    Convert the model at artifact_path to the requested format.
    Returns the path to the exported file in a temp directory.
    Caller is responsible for cleanup (shutil.rmtree on the parent dir).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = Path(artifact_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    logger.info("Loading model from %s for %s export", model_path, export_format)

    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Merge PEFT/LoRA adapters into base weights before export
    if hasattr(model, "merge_and_unload"):
        logger.info("Detected PEFT model — merging LoRA adapters before export")
        model = model.merge_and_unload()

    # Dequantize only if the model is actually quantized (QLoRA / bitsandbytes).
    # transformers.PreTrainedModel always has a dequantize() method but raises
    # ValueError when called on a non-quantized model — guard with is_quantized.
    if getattr(model, "is_quantized", False) and hasattr(model, "dequantize"):
        logger.info("Dequantizing model for export")
        model = model.dequantize()

    model.eval()

    tmp_dir = Path(tempfile.mkdtemp(prefix="modelforge_export_"))

    if export_format == "torchscript":
        return _export_torchscript(model, tokenizer, tmp_dir)

    return _export_onnx(model, tokenizer, tmp_dir, opset_version, optimize)


def _make_dummy_inputs(tokenizer, max_length: int = 128):
    import torch
    dummy = tokenizer(
        "example input text",
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    return dummy


def _export_torchscript(model, tokenizer, tmp_dir: Path) -> Path:
    import torch

    dummy = _make_dummy_inputs(tokenizer)
    out_path = tmp_dir / "model.pt"

    try:
        with torch.no_grad():
            traced = torch.jit.trace(
                model,
                (dummy["input_ids"], dummy["attention_mask"]),
                strict=False,
            )
        torch.jit.save(traced, str(out_path))
        logger.info("TorchScript export successful: %s", out_path)
        return out_path
    except Exception as exc:
        logger.warning("TorchScript trace failed (%s), trying torch.jit.script()", exc)
        try:
            with torch.no_grad():
                scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(out_path))
            logger.info("TorchScript script export successful: %s", out_path)
            return out_path
        except Exception as exc2:
            raise RuntimeError(
                f"TorchScript export failed (trace: {exc}; script: {exc2}). "
                "Consider using ONNX export instead."
            ) from exc2


def _export_onnx(model, tokenizer, tmp_dir: Path, opset_version: int, optimize: bool) -> Path:
    import torch

    # Try optimum first (produces optimized ONNX with graph optimizations)
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        out_dir = tmp_dir / "optimum_onnx"
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model.config._name_or_path,
            export=True,
        )
        ort_model.save_pretrained(str(out_dir))
        onnx_files = list(out_dir.glob("*.onnx"))
        if onnx_files:
            logger.info("Optimum ONNX export successful: %s", onnx_files[0])
            return onnx_files[0]
    except Exception as exc:
        logger.info("optimum not available or failed (%s), using torch.onnx.export fallback", exc)

    # Fallback: torch.onnx.export
    dummy = _make_dummy_inputs(tokenizer)
    out_path = tmp_dir / "model.onnx"

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(out_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids":      {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                },
                opset_version=opset_version,
                do_constant_folding=optimize,
            )
        logger.info("torch.onnx.export successful: %s", out_path)
        return out_path
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc
