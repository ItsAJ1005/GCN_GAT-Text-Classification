import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

"""
Graph Attention Network (GAT) for Document Classification

Key Concepts:
1. Attention Mechanism: Learns to assign different importance to different words
   based on their semantic relationships, regardless of distance.
2. Multi-head Attention: Uses multiple attention heads to capture different types
   of relationships between words (e.g., syntactic, semantic).
3. The model excels at capturing long-range dependencies and non-consecutive word
   relationships that are common in natural language.
"""

class DocumentGAT(nn.Module):
    """
    2-layer Graph Attention Network for document classification.
    
    Input: Batch of document graphs
    - Node features: TF-IDF vectors [num_nodes, vocab_size]
    - Edge indices: Word co-occurrence connections [2, num_edges]
    - Batch vector: Maps nodes to their respective documents
    
    Output: Class probabilities [batch_size, num_classes]
    
    Key Advantages:
    1. Captures long-range dependencies through attention mechanism
    2. Handles non-consecutive word relationships effectively
    3. Learns to focus on important words regardless of their position
    """
    
    def __init__(self, vocab_size, num_classes, hidden_dim=64, num_heads=4, dropout=0.5):
        """
        Initialize the DocumentGAT model.
        
        Args:
            vocab_size (int): Size of the vocabulary (input dimension)
            num_classes (int): Number of output classes
            hidden_dim (int): Hidden dimension size (default: 64)
            num_heads (int): Number of attention heads in first layer (default: 4)
            dropout (float): Dropout rate (default: 0.5)
        """
        super(DocumentGAT, self).__init__()
        
        # First GAT layer with multi-head attention
        # Each head learns different types of word relationships
        self.conv1 = GATConv(
            in_channels=vocab_size,
            out_channels=hidden_dim // num_heads,  # Split hidden_dim across heads
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate head outputs
        )
        
        # Second GAT layer with single attention head
        # Combines information from all previous heads
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=1,  # Single head for final representation
            dropout=dropout,
            concat=False
        )
        
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with xavier uniform and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # GATConv layers handle their own initialization
    
    def forward(self, data):
        """
        Forward pass of the DocumentGAT model.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, vocab_size]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector [num_nodes]
                
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GAT layer: Multi-head attention
        # Each head learns different types of word relationships
        x = F.elu(self.conv1(x, edge_index))  # [num_nodes, hidden_dim]
        x = self.dropout(x)
        
        # Second GAT layer: Combine information from all heads
        # Uses a single attention head to create a unified representation
        x = F.elu(self.conv2(x, edge_index))  # [num_nodes, hidden_dim]
        
        # Global mean pooling: Aggregate node features into document representation
        # The attention mechanism ensures important words contribute more to the final representation
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Final classifier
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)  # Return log probabilities for NLLLoss
