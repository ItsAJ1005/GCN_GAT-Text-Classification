import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from train import evaluate, collate_fn
from torch.utils.data import DataLoader


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_training_history(history, save_dir='.'):
    """Plot training and validation metrics."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def save_classification_report(y_true, y_pred, class_names, save_path):
    """Save classification report to a file."""
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print summary
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def evaluate_model(model, test_loader, criterion, device, class_names, output_dir='results'):
    """
    Evaluate the model and save results.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on
        class_names: List of class names
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate the model
    test_loss, test_acc, test_f1, test_texts, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Print overall metrics
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.json')
    save_classification_report(test_labels, test_preds, class_names, report_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    # Save predictions with text examples
    results = []
    for text, pred, label in zip(test_texts, test_preds, test_labels):
        results.append({
            'text': text,
            'predicted_class': int(pred),
            'true_class': int(label),
            'correct': int(pred) == int(label)
        })
    
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Calculate and save per-class metrics
    metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds, average='weighted'),
        'recall': recall_score(test_labels, test_preds, average='weighted'),
        'f1': f1_score(test_labels, test_preds, average='weighted'),
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        class_mask = (np.array(test_labels) == i)
        if sum(class_mask) > 0:  # Only calculate if class exists in test set
            metrics['per_class'][class_name] = {
                'precision': precision_score(
                    test_labels, test_preds, 
                    labels=[i], average='micro'
                ) if sum(class_mask) > 0 else 0.0,
                'recall': recall_score(
                    test_labels, test_preds, 
                    labels=[i], average='micro'
                ) if sum(class_mask) > 0 else 0.0,
                'f1': f1_score(
                    test_labels, test_preds, 
                    labels=[i], average='micro'
                ) if sum(class_mask) > 0 else 0.0,
                'support': int(sum(class_mask))
            }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nEvaluation results saved to: {os.path.abspath(output_dir)}")
    return metrics


def analyze_misclassifications(predictions_path, output_dir='results'):
    """Analyze and save misclassified examples."""
    with open(predictions_path, 'r') as f:
        results = json.load(f)
    
    misclassified = [r for r in results if not r['correct']]
    
    if not misclassified:
        print("No misclassifications found!")
        return
    
    print(f"\nFound {len(misclassified)} misclassified examples.")
    
    # Group misclassifications by true class and predicted class
    misclass_dict = {}
    for item in misclassified:
        true_class = item['true_class']
        pred_class = item['predicted_class']
        key = (true_class, pred_class)
        if key not in misclass_dict:
            misclass_dict[key] = []
        misclass_dict[key].append(item['text'])
    
    # Save misclassifications to a file
    misclass_path = os.path.join(output_dir, 'misclassifications.txt')
    with open(misclass_path, 'w') as f:
        f.write("Misclassified Examples:\n")
        f.write("=" * 50 + "\n\n")
        
        for (true_class, pred_class), texts in misclass_dict.items():
            f.write(f"True: {true_class} -> Predicted: {pred_class} (Count: {len(texts)})\n")
            f.write("-" * 50 + "\n")
            for i, text in enumerate(texts[:5]):  # Show up to 5 examples per type
                f.write(f"{i+1}. {text}\n")
            if len(texts) > 5:
                f.write(f"... and {len(texts) - 5} more\n")
            f.write("\n")
    
    print(f"Misclassification analysis saved to: {misclass_path}")
