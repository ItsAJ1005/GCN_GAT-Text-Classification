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
nltk.download('punkt')
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
    
    def preprocess_text(self, text, return_string=False):
        """
        Preprocess text by tokenizing, removing stopwords and punctuation.
        
        Args:
            text (str): Input text
            return_string (bool): If True, return preprocessed text as a string
            
        Returns:
            list or str: Processed tokens or joined string
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords, punctuation, and short words
        tokens = [
            word for word in tokens 
            if (word not in self.stop_words and 
                word not in self.punctuation and 
                len(word) > 1 and
                word.isalpha())
        ]
        
        return ' '.join(tokens) if return_string else tokens
    
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
            vocabulary=self.vocab,
            tokenizer=lambda x: x.split(),
            preprocessor=lambda x: x,
            token_pattern=None
        )
        
        # Fit TF-IDF on the preprocessed documents
        preprocessed_docs = [self.preprocess_text(doc, return_string=True) for doc in documents]
        self.vectorizer.fit(preprocessed_docs)
        
        return vocab
    
    def calculate_pmi(self, word1, word2):
        """
        Calculate Pointwise Mutual Information (PMI) between two words.
        
        PMI(i,j) = log(p(i,j) / (p(i) * p(j)))
        where p(i,j) = co-occurrence count / total windows
        
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
            min_pmi (float): Minimum PMI threshold for edge creation
            
        Returns:
            tuple: (node_features, edge_index, edge_weights)
                - node_features: Tensor of shape (num_nodes, vocab_size) with TF-IDF values
                - edge_index: Tensor of shape (2, num_edges) with edge connections
                - edge_weights: Tensor of shape (num_edges,) with PMI values
        """
        # Preprocess text and get tokens
        tokens = self.preprocess_text(text)
        
        # Filter tokens by vocabulary and get unique words
        unique_words = list(set([t for t in tokens if t in self.word_to_idx]))
        num_nodes = len(unique_words)
        
        if num_nodes == 0:
            # Return empty graph with a single node
            return (
                torch.zeros((1, len(self.vocab))),
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0)
            )
        
        # Create node features using TF-IDF
        text_str = ' '.join(tokens)
        tfidf = self.vectorizer.transform([self.preprocess_text(text, return_string=True)])
        node_features = torch.FloatTensor(tfidf.toarray())
        
        # Create edges based on PMI
        edges = []
        edge_weights = []
        
        # Add edges between all pairs of words in the document
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                word1 = unique_words[i]
                word2 = unique_words[j]
                
                # Calculate PMI
                pmi = self.calculate_pmi(word1, word2)
                
                # Add edge if PMI is above threshold
                if pmi > min_pmi:
                    # Add edge in both directions (undirected graph)
                    edges.append((i, j))
                    edges.append((j, i))
                    edge_weights.extend([pmi, pmi])
        
        # Convert to tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_weights = torch.FloatTensor(edge_weights)
        else:
            # If no edges, create self-loops to avoid empty graph
            edge_index = torch.tensor([range(num_nodes), range(num_nodes)], dtype=torch.long)
            edge_weights = torch.ones(num_nodes)
        
        return node_features, edge_index, edge_weights
    
    def texts_to_graphs(self, documents, min_pmi=0.0):
        """
        Convert a list of documents to graph representations.
        
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
