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
    Enhanced 2-layer Graph Attention Network for document classification.
    
    Input: Batch of document graphs
    - Node features: TF-IDF vectors [num_nodes, vocab_size]
    - Edge indices: Word co-occurrence connections [2, num_edges]
    - Batch vector: Maps nodes to their respective documents
    
    Output: Class probabilities [batch_size, num_classes]
    
    Improvements:
    1. Residual connections for better gradient flow
    2. Layer normalization for stable training
    3. Multi-head attention with scaled dot-product attention
    4. Improved feature aggregation
    """
    
    def __init__(self, vocab_size, num_classes, hidden_dim=256, num_heads=8, dropout=0.3):
        """
        Initialize the DocumentGAT model.
        
        Args:
            vocab_size (int): Size of the vocabulary (input dimension)
            num_classes (int): Number of output classes
            hidden_dim (int): Hidden dimension size (default: 256)
            num_heads (int): Number of attention heads in first layer (default: 8)
            dropout (float): Dropout rate (default: 0.3)
        """
        super(DocumentGAT, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(vocab_size, hidden_dim)
        
        # First GAT layer with multi-head attention
        self.conv1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,  # Concatenate head outputs
            add_self_loops=True  # Add self-loops for better message passing
        )
        
        # Layer normalization after first GAT layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Second GAT layer with multi-head attention
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // (num_heads//2),
            heads=num_heads//2,  # Reduce heads in second layer
            dropout=dropout,
            concat=True,
            add_self_loops=True
        )
        
        # Layer normalization after second GAT layer
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combine skip connection
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier head with improved architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
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
        Enhanced forward pass of the DocumentGAT model.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, vocab_size]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector [num_nodes]
                
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial feature projection
        x = self.input_proj(x)
        input_features = x  # Save for residual connection
        
        # First GAT layer with multi-head attention and residual connection
        attn_out = F.elu(self.conv1(x, edge_index))
        attn_out = self.dropout(attn_out)
        x = self.norm1(attn_out + x)  # Residual connection and normalization
        
        # Second GAT layer with reduced heads and residual connection
        attn_out = F.elu(self.conv2(x, edge_index))
        attn_out = self.dropout(attn_out)
        x = self.norm2(attn_out + x)  # Residual connection and normalization
        
        # Global attention pooling
        x_global = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Combine global and local features with skip connection
        x_skip = global_mean_pool(input_features, batch)  # Skip connection from input
        x_combined = torch.cat([x_global, x_skip], dim=-1)  # Concatenate features
        x = self.fusion(x_combined)  # Fuse features
        
        # Final classification with improved head
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)  # Return log probabilities for NLLLoss