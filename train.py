import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch, Data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Import models
from models.gcn import DocumentGCN
from models.gat import DocumentGAT
from data.loader import TextDataset, TextPreprocessor
from data.graph_builder import GraphBuilder

class GraphDataset(Dataset):
        return node_features, edge_index, label, text


def collate_fn(batch):
    """Collate function for creating batches of graphs."""
    node_features_list, edge_indices, labels, texts = zip(*batch)
    
    # Create a batch of graphs
    data_list = []
    for i, (feat, edge_idx, label, text) in enumerate(zip(node_features_list, 
                                                         edge_indices, 
                                                         labels, 
                                                         texts)):
        if edge_idx.size(1) == 0:  # Skip empty graphs
            continue
        data = {
            'x': feat,
            'edge_index': edge_idx,
            'y': torch.tensor([label], dtype=torch.long),
            'text': text
        }
        data_list.append(data)
    
    # Create a batch from the list of graphs
    batch_data = Batch.from_data_list([
        {'x': d['x'], 'edge_index': d['edge_index'], 'y': d['y']} 
        for d in data_list
    ])
    
    # Extract texts and labels for the non-empty graphs
    texts = [d['text'] for d in data_list]
    labels = [d['y'].item() for d in data_list]
    
    return batch_data, torch.tensor(labels, dtype=torch.long), texts


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training"):
        data, labels, _ = batch
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def evaluate(model, loader, criterion, device):
    """Evaluate the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            data, labels, texts = batch
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_texts, all_preds, all_labels


def plot_training_curves(history, model_name, output_dir='results'):
    """Plot training and validation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_training_curves.png'))
    plt.close()

def train_model(model, train_loader, val_loader, test_loader, 
               num_epochs=50, lr=0.01, device='cuda', 
               model_save_path='best_model.pt', patience=10):
    """
    Train and evaluate the model with early stopping.
    
    Args:
        model: The GNN model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        num_epochs: Maximum number of training epochs.
        lr: Learning rate.
        device: Device to train on ('cuda' or 'cpu').
        model_save_path: Path to save the best model.
        patience: Number of epochs to wait before early stopping.
    
    Returns:
        tuple: (best_model, history, test_metrics)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train for one epoch
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_model = model.state_dict()
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for testing
    model.load_state_dict(best_model)
    
    # Evaluate on test set
    test_loss, test_acc, test_f1, test_texts, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    test_metrics = {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': test_f1,
        'classification_report': classification_report(test_labels, test_preds, output_dict=True)
    }
    
    return model, history, test_metrics
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            
            # Early stopping check
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    # Load best model for testing
    model.load_state_dict(torch.load(model_save_path))
    
    # Final evaluation on test set
    test_loss, test_acc, test_f1, test_texts, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Update history with test metrics
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['test_f1'] = test_f1
    
    print(f"\nTest Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    return model, history
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with Val F1: {val_f1:.4f}")
    
    # Load the best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc, test_f1, test_texts, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Save test results
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['test_f1'] = test_f1
    history['test_results'] = {
        'texts': test_texts,
        'preds': test_preds,
        'labels': test_labels
    }
    
    return history
