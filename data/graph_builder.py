import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
from collections import Counter
import torch.nn.functional as F

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class GraphBuilder:
    """
    A class to convert text into graph representations for GNNs.
    """
    def __init__(self, max_nodes=50, window_size=3):
        """
        Initialize the graph builder.
        
        Args:
            max_nodes (int): Maximum number of nodes (words) in the graph.
            window_size (int): Context window size for creating edges between words.
        """
        self.max_nodes = max_nodes
        self.window_size = window_size
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.word_to_idx = {}
        self.vocab = []
    
    def preprocess_text(self, text):
        """
        Preprocess text by tokenizing, removing stopwords and punctuation.
        
        Args:
            text (str): Input text.
            
        Returns:
            list: List of processed tokens.
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords and punctuation
        tokens = [word for word in tokens 
                 if word not in self.stop_words 
                 and word not in self.punctuation]
        return tokens
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts (list): List of text documents.
        """
        # Tokenize all texts and count word frequencies
        all_tokens = []
        for text in texts:
            tokens = self.preprocess_text(text)
            all_tokens.extend(tokens)
        
        # Keep only most frequent words
        word_counts = Counter(all_tokens)
        self.vocab = [word for word, _ in word_counts.most_common(self.max_nodes)]
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
    
    def text_to_graph(self, text):
        """
        Convert a single text into a graph representation.
        
        Args:
            text (str): Input text.
            
        Returns:
            tuple: (node_features, edge_index)
                - node_features: Tensor of shape (num_nodes, embedding_dim)
                - edge_index: Tensor of shape (2, num_edges)
        """
        # Preprocess and filter tokens
        tokens = self.preprocess_text(text)
        tokens = [token for token in tokens if token in self.word_to_idx]
        
        if not tokens:
            # If no valid tokens, return empty graph
            return torch.zeros((1, 300)), torch.zeros((2, 0), dtype=torch.long)
        
        # Limit to max_nodes
        tokens = tokens[:self.max_nodes]
        num_nodes = len(tokens)
        
        # Create node features (using simple one-hot encoding for demonstration)
        # In practice, you might want to use word embeddings here
        node_features = torch.zeros((num_nodes, len(self.vocab)))
        for i, token in enumerate(tokens):
            if token in self.word_to_idx:
                node_features[i, self.word_to_idx[token]] = 1.0
        
        # Create edges based on word co-occurrence within a sliding window
        edges = set()
        for i in range(num_nodes):
            # Connect to words within the window
            for j in range(max(0, i - self.window_size), 
                          min(num_nodes, i + self.window_size + 1)):
                if i != j:
                    edges.add((i, j))
                    edges.add((j, i))  # Undirected graph
        
        # Convert edges to edge_index format
        if edges:
            edge_index = torch.tensor(list(edges), dtype=torch.long).t()
        else:
            # If no edges, create a self-loop to avoid empty graph
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        return node_features, edge_index
    
    def texts_to_graphs(self, texts):
        """
        Convert a list of texts to graph representations.
        
        Args:
            texts (list): List of text documents.
            
        Returns:
            list: List of (node_features, edge_index) tuples.
        """
        self.build_vocabulary(texts)
        graphs = []
        for text in texts:
            graphs.append(self.text_to_graph(text))
        return graphs
