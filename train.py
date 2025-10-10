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
    def __init__(self, texts, labels, graph_builder):
        self.texts = texts
        self.labels = labels
        self.graph_builder = graph_builder
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        node_features, edge_index, edge_weights = self.graph_builder.text_to_graph(text)
        return node_features, edge_index, edge_weights, label, text


def collate_fn(batch):
    """Collate function for creating batches of graphs with edge weights."""
    node_features_list, edge_indices, edge_weights_list, labels, texts = zip(*batch)
    
    # Create a list to store PyG Data objects
    data_list = []
    labels_list = []
    texts_list = []
    
    for i, (feat, edge_idx, edge_weights, label, text) in enumerate(zip(
        node_features_list, 
        edge_indices, 
        edge_weights_list,
        labels, 
        texts
    )):
        if edge_idx.size(1) == 0:  # Skip empty graphs
            continue
            
        # Ensure edge indices are within bounds
        num_nodes = feat.size(0)
        edge_idx = edge_idx.clone()
        edge_idx[edge_idx >= num_nodes] = num_nodes - 1  # Clamp to last node index
        
        # Ensure we have the same number of edges and edge weights
        if edge_weights.size(0) != edge_idx.size(1):
            edge_weights = torch.ones(edge_idx.size(1), device=edge_weights.device)
        
        # Create a PyG Data object
        data = Data(
            x=feat,
            edge_index=edge_idx,
            edge_attr=edge_weights.unsqueeze(1),  # Add edge weights as edge features
            y=torch.tensor([label], dtype=torch.long)
        )
        data_list.append(data)
        labels_list.append(label)
        texts_list.append(text)
    
    if not data_list:  # If all graphs were empty
        return None, None, None
    
    try:
        # Create a batch from the list of Data objects
        batch_data = Batch.from_data_list(data_list)
    except Exception as e:
        print(f"Error creating batch: {e}")
        return None, None, None
    
    return batch_data, torch.tensor(labels_list, dtype=torch.long), texts_list


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        data, labels, _ = batch
        
        # Skip empty batches
        if data is None or labels is None:
            continue
            
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - pass the entire data object to the model
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1
    
    # Calculate metrics
    if num_batches == 0:
        return 0.0, 0.0, 0.0
        
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted') if all_preds else 0.0
    
    return avg_loss, accuracy, f1


def evaluate(model, loader, criterion, device):
    """Evaluate the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_texts = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            data, labels, texts = batch
            
            # Skip empty batches
            if data is None or labels is None:
                continue
                
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass - pass the entire data object to the model
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
            num_batches += 1
    
    # Calculate metrics
    if num_batches == 0:
        return 0.0, 0.0, 0.0, [], [], []
        
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0.0
    f1 = f1_score(all_labels, all_preds, average='weighted') if all_preds else 0.0
    
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
               num_epochs=50, lr=0.0005, device='cuda', 
               model_save_path='best_model.pt', patience=7):
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
        patience: Number of epochs to wait before early stopping (default: 5).
    
    Returns:
        tuple: (best_model, history, test_metrics)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    best_model = None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        model.eval()
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (Val Loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_loss:.4f}")
            break
    
    model.load_state_dict(torch.load(model_save_path))
    
    test_loss, test_acc, test_f1, test_texts, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    test_metrics = {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1': test_f1,
        'classification_report': classification_report(test_labels, test_preds, output_dict=True)
    }
    
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['test_f1'] = test_f1
    
    print(f"\nTest Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Add test results to history
    history['test_results'] = {
        'texts': test_texts,
        'preds': test_preds,
        'labels': test_labels
    }
    
    # Add test metrics to history for easier access
    history['test_metrics'] = test_metrics
    return history
