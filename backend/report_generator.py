# backend/report_generator.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
from typing import Dict, List
import pandas as pd

def generate_training_report(run_id: str) -> Dict:
    """Generate comprehensive training report with charts and metrics"""
    run_dir = Path("runs") / run_id
    
    # Load training data
    metrics_file = run_dir / "metrics.json"
    config_file = run_dir / "config.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError("No metrics found for this run")
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    
    # Generate charts
    charts = _generate_charts(metrics)
    
    # Calculate summary statistics
    summary = _calculate_summary_stats(metrics)
    
    # Generate insights
    insights = _generate_detailed_insights(metrics, config)
    
    report = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "summary": summary,
        "insights": insights,
        "charts": charts,
        "metrics_count": len(metrics),
        "training_duration": summary.get("total_time", 0)
    }
    
    return report

def _generate_charts(metrics: List[Dict]) -> Dict[str, str]:
    """Generate base64 encoded charts"""
    charts = {}
    
    if not metrics:
        return charts
    
    # Extract data
    epochs = [m.get("epoch", i) for i, m in enumerate(metrics)]
    train_losses = [m.get("train_loss") for m in metrics if m.get("train_loss")]
    val_losses = [m.get("val_loss") for m in metrics if m.get("val_loss")]
    train_accs = [m.get("train_accuracy") for m in metrics if m.get("train_accuracy")]
    val_accs = [m.get("val_accuracy") for m in metrics if m.get("val_accuracy")]
    learning_rates = [m.get("learning_rate") for m in metrics if m.get("learning_rate")]
    
    # Loss curves
    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(train_losses)], train_losses, label='Training Loss', color='#ff6b6b')
        if val_losses:
            plt.plot(epochs[:len(val_losses)], val_losses, label='Validation Loss', color='#4ecdc4')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        charts["loss_curve"] = _plot_to_base64()
    
    # Accuracy curves
    if train_accs:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(train_accs)], train_accs, label='Training Accuracy', color='#45b7d1')
        if val_accs:
            plt.plot(epochs[:len(val_accs)], val_accs, label='Validation Accuracy', color='#f9ca24')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        charts["accuracy_curve"] = _plot_to_base64()
    
    # Learning rate schedule
    if learning_rates:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:len(learning_rates)], learning_rates, color='#6c5ce7')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        charts["learning_rate"] = _plot_to_base64()
    
    return charts

def _plot_to_base64() -> str:
    """Convert current matplotlib plot to base64 string"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64

def _calculate_summary_stats(metrics: List[Dict]) -> Dict:
    """Calculate comprehensive summary statistics"""
    if not metrics:
        return {}
    
    train_losses = [m.get("train_loss") for m in metrics if m.get("train_loss")]
    val_losses = [m.get("val_loss") for m in metrics if m.get("val_loss")]
    train_accs = [m.get("train_accuracy") for m in metrics if m.get("train_accuracy")]
    val_accs = [m.get("val_accuracy") for m in metrics if m.get("val_accuracy")]
    
    summary = {
        "total_epochs": len(metrics),
        "best_train_loss": min(train_losses) if train_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "best_train_acc": max(train_accs) if train_accs else None,
        "best_val_acc": max(val_accs) if val_accs else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "final_train_acc": train_accs[-1] if train_accs else None,
        "final_val_acc": val_accs[-1] if val_accs else None,
        "total_time": metrics[-1].get("time_elapsed", 0) if metrics else 0
    }
    
    # Calculate improvement metrics
    if len(train_losses) > 1:
        summary["train_loss_improvement"] = train_losses[0] - train_losses[-1]
    if len(val_losses) > 1:
        summary["val_loss_improvement"] = val_losses[0] - val_losses[-1]
    if len(train_accs) > 1:
        summary["train_acc_improvement"] = train_accs[-1] - train_accs[0]
    if len(val_accs) > 1:
        summary["val_acc_improvement"] = val_accs[-1] - val_accs[0]
    
    return summary

def _generate_detailed_insights(metrics: List[Dict], config: Dict) -> List[str]:
    """Generate detailed training insights"""
    insights = []
    
    if not metrics:
        return ["No training data available for analysis."]
    
    summary = _calculate_summary_stats(metrics)
    
    # Training completion insight
    insights.append(f"Training completed after {summary['total_epochs']} epochs in {summary['total_time']/60:.1f} minutes.")
    
    # Performance insights
    if summary.get("best_val_acc"):
        insights.append(f"Best validation accuracy achieved: {summary['best_val_acc']:.3f}")
    
    if summary.get("val_loss_improvement"):
        if summary["val_loss_improvement"] > 0:
            insights.append(f"Validation loss improved by {summary['val_loss_improvement']:.4f} during training.")
        else:
            insights.append("Validation loss increased during training, indicating potential overfitting.")
    
    # Overfitting detection
    if summary.get("final_train_acc") and summary.get("final_val_acc"):
        gap = summary["final_train_acc"] - summary["final_val_acc"]
        if gap > 0.1:
            insights.append(f"Large train-validation accuracy gap ({gap:.3f}) suggests overfitting.")
        elif gap < 0.05:
            insights.append("Good generalization with minimal overfitting detected.")
    
    # Model configuration insights
    if config.get("learning_rate"):
        insights.append(f"Training used learning rate of {config['learning_rate']:.2e}")
    
    if config.get("batch_size"):
        insights.append(f"Batch size: {config['batch_size']}")
    
    return insights

def generate_model_package(run_id: str) -> Path:
    """Generate complete model package with inference code"""
    run_dir = Path("runs") / run_id
    package_dir = Path("exports") / f"{run_id}_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    model_files = ["pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    for file_name in model_files:
        src_file = run_dir / file_name
        if src_file.exists():
            import shutil
            shutil.copy2(src_file, package_dir / file_name)
    
    # Generate inference script
    inference_code = _generate_inference_script(run_id)
    (package_dir / "inference.py").write_text(inference_code)
    
    # Generate requirements.txt
    requirements = """torch>=1.9.0
transformers>=4.20.0
numpy>=1.21.0
pandas>=1.3.0
"""
    (package_dir / "requirements.txt").write_text(requirements)
    
    # Generate README
    readme = _generate_readme(run_id)
    (package_dir / "README.md").write_text(readme)
    
    # Create zip file
    import shutil
    zip_path = Path("exports") / f"{run_id}_complete_package"
    shutil.make_archive(str(zip_path), 'zip', package_dir)
    
    return Path(f"{zip_path}.zip")

def _generate_inference_script(run_id: str) -> str:
    """Generate ready-to-use inference script"""
    return f'''#!/usr/bin/env python3
"""
Inference script for model {run_id}
Generated by No-Code ML Training Platform
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

class ModelInference:
    def __init__(self, model_path="."):
        """Initialize the model for inference"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping if available
        self.id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else None
        
    def predict(self, text: str, return_probabilities: bool = False):
        """
        Predict class for input text
        
        Args:
            text (str): Input text to classify
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        result = {{
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predicted_label": self.id2label.get(predicted_class, f"Class_{predicted_class}") if self.id2label else f"Class_{predicted_class}"
        }}
        
        if return_probabilities:
            result["all_probabilities"] = {{
                self.id2label.get(i, f"Class_{{i}}") if self.id2label else f"Class_{{i}}": prob.item()
                for i, prob in enumerate(probabilities[0])
            }}
            
        return result
    
    def predict_batch(self, texts: list, batch_size: int = 32):
        """
        Predict classes for multiple texts
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
            
            # Process results
            for j, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
                confidence = probs[pred_class].item()
                results.append({{
                    "text": batch_texts[j],
                    "predicted_class": pred_class.item(),
                    "confidence": confidence,
                    "predicted_label": self.id2label.get(pred_class.item(), f"Class_{{pred_class.item()}}") if self.id2label else f"Class_{{pred_class.item()}}"
                }})
                
        return results

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = ModelInference()
    
    # Single prediction
    text = "This is a sample text for classification"
    result = model.predict(text, return_probabilities=True)
    print("Single prediction:")
    print(json.dumps(result, indent=2))
    
    # Batch prediction
    texts = [
        "First sample text",
        "Second sample text", 
        "Third sample text"
    ]
    results = model.predict_batch(texts)
    print("\\nBatch predictions:")
    for result in results:
        print(f"Text: {{result['text'][:50]}}...")
        print(f"Prediction: {{result['predicted_label']}} ({{result['confidence']:.3f}})")
        print()
'''

def _generate_readme(run_id: str) -> str:
    """Generate comprehensive README for model package"""
    return f'''# Model Package: {run_id}

This package contains a trained model and inference code generated by the No-Code ML Training Platform.

## Contents

- `pytorch_model.bin` - Trained model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `inference.py` - Ready-to-use inference script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run inference:
```python
from inference import ModelInference

# Initialize model
model = ModelInference()

# Make prediction
result = model.predict("Your text here")
print(result)
```

## API Reference

### ModelInference Class

#### `__init__(model_path=".")`
Initialize the model for inference.

#### `predict(text, return_probabilities=False)`
Predict class for a single text input.

**Parameters:**
- `text` (str): Input text to classify
- `return_probabilities` (bool): Whether to return all class probabilities

**Returns:**
- `dict`: Prediction results with class, confidence, and label

#### `predict_batch(texts, batch_size=32)`
Predict classes for multiple texts efficiently.

**Parameters:**
- `texts` (list): List of input texts
- `batch_size` (int): Batch size for processing

**Returns:**
- `list`: List of prediction results

## Model Information

- **Model ID:** {run_id}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Framework:** PyTorch + Transformers

## Support

For questions or issues, refer to the No-Code ML Training Platform documentation.
'''