import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

class GCN(nn.Module):
    """
    Graph Convolutional Network for text classification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        """
        Initialize the GCN model.
        
        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(GCN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with xavier uniform and zeros for biases."""
        for m in self.modules():
            if isinstance(m, (GCNConv, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GCN model.
        
        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, input_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            batch (torch.Tensor, optional): Batch vector [num_nodes] which maps each node to its graph.
                                          If None, assumes all nodes belong to the same graph.
        
        Returns:
            torch.Tensor: Output logits [batch_size, output_dim].
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global max pooling to get graph-level representation
        if batch is not None:
            x = global_max_pool(x, batch)  # [batch_size, hidden_dim]
        else:
            x = x.max(dim=0, keepdim=True)[0]  # [1, hidden_dim] if single graph
        
        # Final classifier
        x = self.fc(x)
        
        return x
