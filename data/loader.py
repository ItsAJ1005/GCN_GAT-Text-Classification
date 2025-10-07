import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data

class TextDataset:
    """
    A class to load and preprocess text datasets for graph-based classification.
    """
    def __init__(self, data_path=None, test_size=0.2, val_size=0.1, random_state=42):
        """
        Initialize the dataset loader.
        
        Args:
            data_path (str, optional): Path to the dataset file. If None, uses a sample dataset.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training set to include in the validation split.
            random_state (int): Random seed for reproducibility.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.data = None
        self.labels = None
        self.label_map = {}
        
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
        else:
            self._create_sample_data()
    
    def load_data(self, file_path):
        """
        Load data from a CSV file with 'text' and 'label' columns.
        
        Args:
            file_path (str): Path to the CSV file.
        """
        df = pd.read_csv(file_path)
        self.texts = df['text'].tolist()
        
        # Create label mapping
        unique_labels = sorted(set(df['label']))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.labels = [self.label_map[label] for label in df['label']]
    
    def _create_sample_data(self):
        """Create a small sample dataset for testing purposes."""
        self.texts = [
            "This is a positive review. I really enjoyed the movie!",
            "Negative review. The film was terrible and boring.",
            "The product works as expected, good quality.",
            "Not worth the money, very disappointed with the purchase.",
            "Amazing experience, would definitely recommend to others.",
            "Poor customer service and bad quality product."
        ]
        self.labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
        self.label_map = {0: "negative", 1: "positive"}
    
    def get_splits(self):
        """
        Split the dataset into train, validation, and test sets.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate out test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.texts, self.labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.labels
        )
        
        # Second split: split remaining data into train and validation
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_num_classes(self):
        """Get the number of unique classes in the dataset."""
        return len(self.label_map)

    def get_label_map(self):
        """Get the mapping from label indices to label names."""
        return {v: k for k, v in self.label_map.items()}


def create_pyg_data(text, label, edge_index, node_features):
    """
    Create a PyTorch Geometric Data object from text and graph information.
    
    Args:
        text (str): The input text.
        label (int): The class label.
        edge_index (torch.Tensor): Graph connectivity in COO format (2, num_edges).
        node_features (torch.Tensor): Node feature matrix (num_nodes, feature_dim).
        
    Returns:
        torch_geometric.data.Data: A PyG Data object.
    """
    return Data(
        x=node_features,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
        text=text
    )
