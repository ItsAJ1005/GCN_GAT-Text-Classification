import os
import sys
import torch
from data.loader import TextDataset
from data.graph_builder import GraphBuilder

def test_graph_builder(dataset_name='mr', num_samples=5, min_pmi=0.1):
    """
    Test the GraphBuilder with a sample dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('r8' or 'mr')
        num_samples (int): Number of sample graphs to visualize
        min_pmi (float): Minimum PMI threshold for edge creation
    """
    print(f"\n{'='*50}")
    print(f"Testing GraphBuilder with {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Load the dataset
    dataset = TextDataset(dataset_name=dataset_name, preprocess=True)
    train_df, test_df = dataset.get_splits()
    
    # Get a sample of documents
    sample_docs = train_df['text'].tolist()[:num_samples]
    
    # Initialize the graph builder
    graph_builder = GraphBuilder(
        min_word_freq=5,  # Minimum word frequency
        window_size=3,    # Context window size
        max_vocab_size=10000  # Maximum vocabulary size
    )
    
    # Build vocabulary and co-occurrence matrix
    print("\nBuilding vocabulary and co-occurrence matrix...")
    graph_builder.build_vocab(train_df['text'].tolist())
    graph_builder._get_cooccurrence_matrix(train_df['text'].tolist())
    
    print(f"\nVocabulary size: {len(graph_builder.vocab)}")
    print(f"Number of documents: {len(train_df)}")
    
    # Process and visualize sample documents
    for i, doc in enumerate(sample_docs):
        print(f"\nProcessing document {i+1}:")
        print("-" * 50)
        print(f"Original text: {doc[:200]}..." if len(doc) > 200 else doc)
        
        # Convert text to graph
        node_features, edge_index, edge_weights = graph_builder.text_to_graph(doc, min_pmi=min_pmi)
        
        # Print graph statistics
        print(f"\nGraph statistics:")
        print(f"- Number of nodes: {node_features.size(0)}")
        print(f"- Number of edges: {edge_index.size(1) // 2}")  # Divide by 2 for undirected graph
        print(f"- Node feature dimension: {node_features.size(1)}")
        
        # Visualize the graph (only if not too large)
        if node_features.size(0) <= 20:  # Only visualize small graphs
            print("\nVisualizing graph...")
            graph_builder.visualize_graph(
                (node_features, edge_index, edge_weights),
                max_nodes=20,
                figsize=(10, 8)
            )
        else:
            print("\nGraph is too large for visualization. Try with a shorter document.")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # Test with both datasets
    for dataset in ['mr', 'r8']:
        test_graph_builder(dataset_name=dataset, num_samples=2, min_pmi=0.1)
    
    print("Testing completed!")
