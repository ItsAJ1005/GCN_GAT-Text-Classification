import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

"""
Simple 2-layer GCN for Document Classification

Key Concepts:
1. Graph Convolution: Aggregates information from neighboring nodes to learn
   meaningful node representations that capture local graph structure.
2. Global Pooling: Converts variable-sized graphs into fixed-size document
   representations by aggregating node features.
3. The model learns both local word relationships (via GCN) and global document
   semantics (via pooling) for classification.
"""

class DocumentGCN(nn.Module):
    """
    2-layer GCN for document classification.
    
    Input: Batch of document graphs
    - Node features: TF-IDF vectors [num_nodes, vocab_size]
    - Edge indices: Word co-occurrence connections [2, num_edges]
    - Batch vector: Maps nodes to their respective documents
    
    Output: Class probabilities [batch_size, num_classes]
    """
    def __init__(self, vocab_size, num_classes, hidden_dim=64, dropout=0.5):
        """
        Initialize the DocumentGCN model.
        
        Args:
            vocab_size (int): Size of the vocabulary (input dimension)
            num_classes (int): Number of output classes
            hidden_dim (int): Hidden dimension size (default: 64)
            dropout (float): Dropout rate (default: 0.5)
        """
        super(DocumentGCN, self).__init__()
        
        # First GCN layer: Transform TF-IDF features to hidden representation
        # This captures local word relationships within the document
        self.conv1 = GCNConv(vocab_size, hidden_dim)
        
        # Second GCN layer: Refine node representations
        # Deeper layer captures higher-order relationships between words
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier head: Maps document embedding to class scores
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
    
    def forward(self, data):
        """
        Forward pass of the DocumentGCN model.
        
        Args:
            data: PyG Data object containing:
                - x: Node features [num_nodes, vocab_size]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector [num_nodes]
                
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GCN layer: Local feature transformation
        # Each node aggregates information from its neighbors
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer: Higher-level feature learning
        # Nodes now have access to information from their 2-hop neighborhood
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global mean pooling: Convert node embeddings to document embedding
        # This is crucial as it aggregates all word-level information
        # into a single vector representing the entire document
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Final classifier: Map document embedding to class scores
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)  # Return log probabilities for NLLLoss
