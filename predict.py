import torch
import torch.nn.functional as F
from models.gcn import DocumentGCN  # Updated import path
from models.gat import DocumentGAT  # For GAT model if needed
from torch_geometric.data import Data
import pickle
import os
import numpy as np
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import sys

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

# Ensure NLTK data is downloaded
download_nltk_data()

class TextClassifier:
    def __init__(self, model_path, vocab_path, class_names, device='cpu', model_type='gat'):
        """
        Initialize the text classifier.
        
        Args:
            model_path: Path to the trained model
            vocab_path: Path to the saved vocabulary
            class_names: List of class names
            device: Device to run inference on
            model_type: Type of model ('gcn' or 'gat')
        """
        self.device = device
        self.class_names = class_names
        self.model_type = model_type.lower()
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            
        # Handle different vocabulary formats
        if isinstance(vocab_data, dict) and 'word_to_idx' in vocab_data:
            # New format with word_to_idx and vocab
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
            self.vocab = vocab_data.get('vocab', list(self.word_to_idx.keys()))
        elif isinstance(vocab_data, dict):
            # Assume it's just the word_to_idx dict
            self.word_to_idx = vocab_data
            self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
            self.vocab = list(self.word_to_idx.keys())
        else:
            # It's a list of words
            self.vocab = vocab_data
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        vocab_size = len(self.word_to_idx)
        num_classes = len(class_names)
        hidden_dim = 128  # Should match your training config
        
        print(f"Loading {self.model_type.upper()} model with vocab size: {vocab_size}, {num_classes} classes")
        
        # Initialize the appropriate model
        if self.model_type == 'gat':
            self.model = DocumentGAT(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_heads=4,  # Default value, adjust if different in your training
                dropout=0.5   # Default value, adjust if different in your training
            )
        else:  # Default to GCN
            self.model = DocumentGCN(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes
            )
        
        # Load the trained weights
        state_dict = torch.load(model_path, map_location=torch.device(device))
        
        # Handle potential mismatch in state dict keys
        model_state_dict = self.model.state_dict()
        
        # Filter out unnecessary keys and handle size mismatches
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if k in model_state_dict and v.size() == model_state_dict[k].size()}
        
        # Update the model's state dict with the filtered state dict
        model_state_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Initialize text preprocessing
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess text: tokenize, remove stopwords, lemmatize."""
        if not isinstance(text, str):
            return []
            
        try:
            # Simple tokenization if NLTK fails
            tokens = word_tokenize(text.lower())
            
            # Filter and lemmatize
            processed_tokens = []
            for token in tokens:
                if token.isalnum() and token not in self.stop_words:
                    try:
                        # Try lemmatization
                        lemma = self.lemmatizer.lemmatize(token)
                        processed_tokens.append(lemma)
                    except:
                        # If lemmatization fails, use the original token
                        processed_tokens.append(token)
            
            return processed_tokens
            
        except Exception as e:
            print(f"Warning: Error in text preprocessing: {str(e)}")
            # Fallback to simple whitespace splitting
            return [t for t in text.lower().split() if t.isalnum()]
    
    def text_to_indices(self, text, max_length=100):
        """Convert text to vocabulary indices."""
        tokens = self.preprocess_text(text)[:max_length]
        unk_idx = self.word_to_idx.get('<unk>', 0)
        return [self.word_to_idx.get(token, unk_idx) for token in tokens]
    
    def predict(self, text):
        """Predict the class of a single text."""
        # Convert text to indices
        indices = self.text_to_indices(text)
        if not indices:
            return {
                'class': 'unknown',
                'confidence': 0.0,
                'probabilities': {cls: 0.0 for cls in self.class_names}
            }
        
        # Create bag-of-words representation
        x = torch.zeros((1, len(self.vocab)), dtype=torch.float).to(self.device)
        for idx in indices:
            x[0, idx] += 1
        
        # Create a simple graph (single node for now)
        edge_index = torch.tensor([[], []], dtype=torch.long, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(data)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        return {
            'class': self.class_names[pred_class],
            'confidence': probs[0][pred_class].item(),
            'probabilities': {self.class_names[i]: prob.item() for i, prob in enumerate(probs[0])}
        }

def load_classifier(device='cpu', model_type='gat', dataset='mr'):
    """Helper function to load the classifier using organized directory structure.
    
    Args:
        device: Device to run the model on ('cpu' or 'cuda')
        model_type: Type of model ('gcn' or 'gat')
        dataset: Name of the dataset ('mr', '20news', etc.)
    """
    # Use organized directory structure: models/{dataset}/{model_type}/
    model_dir = Path('models') / dataset.lower() / model_type.lower()
    model_path = model_dir / 'best_model.pt'
    vocab_path = model_dir / 'vocab.pkl'
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please train the model first using:\n"
            f"  python main.py --dataset {dataset} --model-type {model_type}"
        )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_path}\n"
            f"Please train the model first to generate the vocabulary."
        )
    
    # Set class names based on dataset
    if dataset.lower() == 'mr':
        class_names = ['negative', 'positive']  # MR dataset has 2 classes
    else:  # 20 Newsgroups - all 20 classes
        class_names = [
            'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
            'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
            'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
            'sci.space', 'soc.religion.christian', 'talk.politics.guns',
            'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
        ]
    
    print(f"Loading {model_type.upper()} model for {dataset} dataset...")
    print(f"Model directory: {model_dir}")
    
    return TextClassifier(str(model_path), str(vocab_path), class_names, device, model_type=model_type)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run text classification prediction')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--dataset', type=str, default='mr', choices=['mr', '20news'],
                       help='Dataset name (mr or 20news)')
    parser.add_argument('--model-type', type=str, default='gat', choices=['gcn', 'gat'], 
                       help='Model type (gcn or gat)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    args = parser.parse_args()
    
    try:
        print("Loading classifier...")
        classifier = load_classifier(
            device=args.device,
            model_type=args.model_type,
            dataset=args.dataset
        )
        print("Classifier loaded successfully!")
        
        if args.text:
            # Classify the provided text
            result = classifier.predict(args.text)
            print("\nPrediction Result:")
            print("-" * 50)
            print(f"Text: {args.text}")
            print(f"Predicted class: {result['class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Class probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
            print("-" * 50)
        else:
            # Default test texts based on dataset
            if args.dataset.lower() == 'mr':
                test_texts = [
                    "This movie was great! I really enjoyed it.",
                    "The film was terrible and boring.",
                    "The acting was superb and the plot was engaging.",
                    "I wouldn't recommend this movie to anyone."
                ]
            else:  # 20 Newsgroups
                test_texts = [
                    "The new graphics card has amazing performance for gaming",
                    "The baseball game was intense last night with a home run in the 9th inning",
                    "NASA announced a new mission to explore Mars in 2024",
                    "The peace talks in the Middle East are showing promising progress"
                ]
            
            print("\nTest Predictions:")
            print("-" * 50)
            for text in test_texts:
                result = classifier.predict(text)
                print(f"\nText: {text}")
                print(f"Predicted class: {result['class']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("Class probabilities:")
                for cls, prob in result['probabilities'].items():
                    print(f"  {cls}: {prob:.4f}")
                print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
        sys.exit(1)



