import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from models.gcn import DocumentGCN
from data.loader import TextDataset
from data.graph_builder import GraphBuilder

def test_document_gcn():
    """
    Test the DocumentGCN model with a sample document.
    """
    print("Testing DocumentGCN model...")
    print("=" * 50)
    
    # Load sample dataset
    dataset = TextDataset(dataset_name='mr', preprocess=True)
    train_df, test_df = dataset.get_splits()
    
    # Initialize graph builder and build vocabulary
    graph_builder = GraphBuilder(min_word_freq=5, window_size=3)
    graph_builder.build_vocab(train_df['text'].tolist())
    graph_builder._get_cooccurrence_matrix(train_df['text'].tolist())
    
    # Get a sample document and its label
    sample_text = train_df.iloc[0]['text']
    sample_label = train_df.iloc[0]['label_id']
    
    print(f"Sample document (first 200 chars):\n{sample_text[:200]}...")
    print(f"\nTrue label: {dataset.get_label_name(sample_label)}")
    
    # Convert document to graph
    node_features, edge_index, edge_weights = graph_builder.text_to_graph(sample_text)
    
    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_weights,
        y=torch.tensor([sample_label], dtype=torch.long),
        num_nodes=node_features.size(0)
    )
    
    # Create a DataLoader (batch size = 1 for this example)
    loader = DataLoader([data], batch_size=1, shuffle=False)
    
    # Initialize model
    vocab_size = len(graph_builder.vocab)
    num_classes = dataset.get_num_classes()
    model = DocumentGCN(
        vocab_size=vocab_size,
        num_classes=num_classes,
        hidden_dim=64,
        dropout=0.5
    )
    
    print("\nModel architecture:")
    print(model)
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Get model predictions
            log_probs = model(batch)
            probs = torch.exp(log_probs)
            
            # Get predicted class
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            
            print(f"\nPredicted class: {dataset.get_label_name(pred_class)} (confidence: {confidence:.2f})")
            
            # Print class probabilities
            print("\nClass probabilities:")
            for i, prob in enumerate(probs.squeeze()):
                print(f"- {dataset.get_label_name(i)}: {prob:.4f}")

if __name__ == "__main__":
    test_document_gcn()
