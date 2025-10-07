import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams, FreqDist
import nltk
import string
from collections import defaultdict, Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class GraphBuilder:
    """
    A class to convert text documents into graph-of-words representations.
    Implements PMI-based edge weighting and provides visualization capabilities.
    """
    
    def __init__(self, min_word_freq=5, window_size=3, max_vocab_size=10000):
        """
        Initialize the graph builder.
        
        Args:
            min_word_freq (int): Minimum word frequency to include in vocabulary
            window_size (int): Context window size for co-occurrence counting
            max_vocab_size (int): Maximum size of the vocabulary
        """
        self.min_word_freq = min_word_freq
        self.window_size = window_size
        self.max_vocab_size = max_vocab_size
        
        # Text processing
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        
        # Vocabulary and statistics
        self.vocab = []
        self.word_to_idx = {}
        self.word_freq = Counter()
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self.total_windows = 0
        
        # PMI cache
        self.pmi_cache = {}
        
        # For TF-IDF node features
        self.vectorizer = None
        
    def preprocess_text(self, text):
        """
        Preprocess a single text document.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of processed tokens
        """
        if not isinstance(text, str):
            text = str(text)
            
        try:
            # Simple whitespace tokenization as fallback
            tokens = text.lower().split()
            
            # Basic cleaning
            tokens = [
                word.strip(''.join(self.punctuation))
                for word in tokens
                if word.strip(''.join(self.punctuation))  # Remove empty strings
            ]
            
            # Remove stopwords and short tokens
            tokens = [
                word 
                for word in tokens 
                if word not in self.stop_words 
                and len(word) > 1
            ]
            
            return tokens
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return []  # Return empty list for problematic texts
            
    def build_vocab(self, documents):
        """
        Build vocabulary from a list of documents.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            dict: Vocabulary mapping words to indices
        """
        # Count word frequencies across all documents
        word_counts = Counter()
        for doc in documents:
            tokens = self.preprocess_text(doc)
            word_counts.update(tokens)
        
        # Filter by minimum frequency and limit vocab size
        vocab = {
            word: idx for idx, (word, count) in enumerate(
                sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
            ) if count >= self.min_word_freq
        }
        
        # Limit vocabulary size
        if len(vocab) > self.max_vocab_size:
            vocab = dict(list(vocab.items())[:self.max_vocab_size])
        
        # Add special tokens
        vocab['<PAD>'] = len(vocab)
        vocab['<UNK>'] = len(vocab)
        
        self.vocab = list(vocab.keys())
        self.word_to_idx = vocab
        self.idx_to_word = {v: k for k, v in vocab.items()}
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,  # Already tokenized
            preprocessor=None,  # Already preprocessed
            token_pattern=None
        )
        
        # Convert token lists to space-separated strings for TF-IDF
        preprocessed_docs = [' '.join(self.preprocess_text(doc)) for doc in documents]
        self.vectorizer.fit(preprocessed_docs)
        
        return self.word_to_idx
        
    def calculate_pmi(self, word1, word2):
        """
        Calculate Pointwise Mutual Information (PMI) between two words.
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            float: PMI score between word1 and word2
        """
        # Check cache first
        if (word1, word2) in self.pmi_cache:
            return self.pmi_cache[(word1, word2)]
        
        # Get word frequencies and co-occurrence count
        count_i = self.word_freq[word1]
        count_j = self.word_freq[word2]
        count_ij = self.cooccurrence[word1][word2]
        
        # Calculate probabilities
        p_i = count_i / self.total_windows
        p_j = count_j / self.total_windows
        p_ij = count_ij / self.total_windows
        
        # Calculate PMI (with smoothing to avoid log(0))
        if p_ij > 0 and p_i > 0 and p_j > 0:
            pmi = math.log(p_ij / (p_i * p_j))
        else:
            pmi = 0.0
        
        # Cache the result
        self.pmi_cache[(word1, word2)] = pmi
        self.pmi_cache[(word2, word1)] = pmi  # PMI is symmetric
        
        return pmi
    
    def _get_cooccurrence_matrix(self, documents):
        """
        Build co-occurrence matrix from documents.
        
        Args:
            documents (list): List of text documents
        """
        # Reset counters
        self.word_freq = Counter()
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self.total_windows = 0
        
        for doc in documents:
            tokens = self.preprocess_text(doc)
            
            # Only keep words in vocabulary
            tokens = [token for token in tokens if token in self.word_to_idx]
            
            # Update word frequencies
            self.word_freq.update(tokens)
            
            # Update co-occurrence counts using sliding window
            for i in range(len(tokens)):
                start = max(0, i - self.window_size)
                end = min(len(tokens), i + self.window_size + 1)
                
                # All words in the window co-occur with the center word
                for j in range(start, end):
                    if i != j:  # Skip self-loops for now
                        word1 = tokens[i]
                        word2 = tokens[j]
                        self.cooccurrence[word1][word2] += 1
                        self.total_windows += 1
    
    def text_to_graph(self, text, min_pmi=0.0):
        """
        Convert a single text document into a graph representation.
        
        Args:
            text (str): Input text document
            min_pmi (float): Not used, kept for backward compatibility
            
        Returns:
            tuple: (node_features, edge_index, edge_weights)
                - node_features: Tensor of shape (num_nodes, vocab_size) with one-hot encoded word indices
                - edge_index: Tensor of shape (2, num_edges) with edge connections
                - edge_weights: Tensor of shape (num_edges,) with edge weights based on co-occurrence
        """
        # Preprocess text and get tokens
        tokens = self.preprocess_text(text)
        
        # Get unique words in the document that are in our vocabulary
        unique_words = list(set([word for word in tokens if word in self.word_to_idx]))
        num_nodes = len(unique_words)
        
        if num_nodes == 0:
            # Return empty graph with a single node
            return (
                torch.zeros((1, len(self.vocab))),  # One node with zero features
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0)
            )
        
        # Create one-hot encoded node features
        node_features = torch.zeros((num_nodes, len(self.vocab)), dtype=torch.float)
        for i, word in enumerate(unique_words):
            if word in self.word_to_idx:
                node_features[i, self.word_to_idx[word]] = 1.0
        
        # Create edges based on co-occurrence within a window
        edges = []
        edge_weights = []
        
        # Create a local co-occurrence matrix for this document
        local_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Build local co-occurrence matrix
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + self.window_size + 1, len(tokens))):
                word1 = tokens[i]
                word2 = tokens[j]
                
                # Only consider words in our vocabulary
                if word1 in self.word_to_idx and word2 in self.word_to_idx:
                    local_cooccurrence[word1][word2] += 1
                    local_cooccurrence[word2][word1] += 1  # Undirected graph
        
        # Create a mapping from word to node index
        word_to_node = {word: i for i, word in enumerate(unique_words)}
        
        # Add edges based on local co-occurrence
        for word1 in local_cooccurrence:
            for word2, count in local_cooccurrence[word1].items():
                if count > 0 and word1 in word_to_node and word2 in word_to_node:
                    i = word_to_node[word1]
                    j = word_to_node[word2]
                    
                    # Add edge in both directions for undirected graph
                    edges.append([i, j])
                    edges.append([j, i])
                    
                    # Use log(1 + count) as edge weight to reduce the impact of high-frequency pairs
                    weight = math.log(1 + count)
                    edge_weights.append(weight)
                    edge_weights.append(weight)
        
        if not edges:
            # If no edges were added, add self-loops to all nodes
            edges = [[i, i] for i in range(num_nodes)]
            edge_weights = [1.0] * num_nodes
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return node_features, edge_index, edge_weights
    
    def texts_to_graphs(self, documents, min_pmi=0.0):
        """
{{ ... }}
        
        Args:
            documents (list): List of text documents
            min_pmi (float): Minimum PMI threshold for edge creation
            
        Returns:
            list: List of (node_features, edge_index, edge_weights) tuples
        """
        # First build vocabulary and co-occurrence statistics
        self.build_vocab(documents)
        self._get_cooccurrence_matrix(documents)
        
        # Then convert each document to a graph
        graphs = []
        for doc in documents:
            graphs.append(self.text_to_graph(doc, min_pmi=min_pmi))
            
        return graphs
    
    def visualize_graph(self, graph, max_nodes=20, figsize=(12, 8)):
        """
        Visualize a graph using NetworkX and Matplotlib.
        
        Args:
            graph (tuple): Tuple of (node_features, edge_index, edge_weights)
            max_nodes (int): Maximum number of nodes to display
            figsize (tuple): Figure size
        """
        node_features, edge_index, edge_weights = graph
        
        # Limit number of nodes for visualization
        num_nodes = min(node_features.size(0), max_nodes)
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes with their corresponding words
        for i in range(num_nodes):
            # Get the word with highest TF-IDF score for this node
            word_idx = node_features[i].argmax().item()
            word = self.idx_to_word.get(word_idx, f"word_{word_idx}")
            G.add_node(i, label=word)
        
        # Add edges with PMI weights
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src < num_nodes and tgt < num_nodes:  # Only include visible nodes
                weight = edge_weights[i].item()
                G.add_edge(src, tgt, weight=weight)
        
        # Draw the graph
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)  # For consistent layout
        
        # Get node labels and edge weights
        labels = nx.get_node_attributes(G, 'label')
        weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]  # Scale for visibility
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # Add edge labels (PMI values)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Text Graph Visualization (Node: word, Edge: PMI)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
