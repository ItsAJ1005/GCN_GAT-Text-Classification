import os
import re
import numpy as np
import pandas as pd
import requests
import tarfile
import io
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def download_file(url, filename, data_dir='data'):
    """Download a file from URL and save it locally."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return filepath

def extract_tar_gz(filepath, extract_to='data'):
    """Extract a .tar.gz file."""
    print(f"Extracting {filepath}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=extract_to)

class TextPreprocessor:
    """Text preprocessing utilities."""
    def __init__(self, remove_stopwords=True):
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
        self.pattern = re.compile(r'[^\w\s]')
    
    def clean_text(self, text):
        """Clean and preprocess text."""
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = self.pattern.sub(' ', text)
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords and short tokens
            tokens = [word for word in tokens 
                     if word not in self.stopwords 
                     and len(word) > 1]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error processing text: {e}")
            return ''  # Return empty string for problematic texts

class TextDataset:
    """A class to load and preprocess text classification datasets."""
    
    DATASETS = {
        '20news': {
            'categories': ['sci.space', 'comp.graphics', 'rec.sport.baseball', 'talk.politics.mideast'],
            'subset': 'all',  # 'all', 'train', or 'test'
            'shuffle': True,
            'random_state': 42,
            'remove': ('headers', 'footers', 'quotes')  # Remove headers, footers, and quotes
        },
        'mr': {
            'url': 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz',
            'filename': 'mr.tar.gz',
            'extract_dir': 'movie_review',
            'pos_file': 'rt-polarity.pos',
            'neg_file': 'rt-polarity.neg'
        }
    }
    
    def __init__(self, dataset_name='r8', data_dir='data', test_size=0.2, 
                 random_state=42, preprocess=True):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name (str): Name of the dataset ('r8' or 'mr').
            data_dir (str): Directory to store the dataset.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
            preprocess (bool): Whether to preprocess the text.
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = TextPreprocessor() if preprocess else None
        self.df = None
        self.label_map = {}
        
        if self.dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(self.DATASETS.keys())}")
        
        # Load and preprocess the dataset
        self._load_dataset()
        self._print_statistics()
    
    def _download_dataset(self):
        """Download and extract the dataset if not already present."""
        dataset_info = self.DATASETS[self.dataset_name]
        os.makedirs(self.data_dir, exist_ok=True)
        
        # For R8 dataset, we need to download both train and test files
        if self.dataset_name == 'r8':
            train_file = os.path.join(self.data_dir, dataset_info['filename'])
            test_file = os.path.join(self.data_dir, dataset_info['test_filename'])
            
            # Download train file if it doesn't exist
            if not os.path.exists(train_file):
                download_file(dataset_info['url'], dataset_info['filename'], self.data_dir)
                
            # Download test file if it doesn't exist
            if not os.path.exists(test_file):
                download_file(dataset_info['test_url'], dataset_info['test_filename'], self.data_dir)
        else:
            # For other datasets (like MR), use the original logic
            filepath = os.path.join(self.data_dir, dataset_info['filename'])
            if not os.path.exists(filepath):
                filepath = download_file(dataset_info['url'], dataset_info['filename'], self.data_dir)
    def _load_20news(self):
        """Load the 20 Newsgroups dataset."""
        dataset_info = self.DATASETS['20news']
        
        # Load the dataset
        newsgroups = fetch_20newsgroups(
            subset=dataset_info['subset'],
            categories=dataset_info['categories'],
            shuffle=dataset_info['shuffle'],
            random_state=dataset_info['random_state'],
            remove=dataset_info['remove']
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': newsgroups.data,
            'label': [newsgroups.target_names[t] for t in newsgroups.target]
        })
        
        return df
        
    def _load_mr(self):
        """Load the MR dataset."""
        dataset_info = self.DATASETS['mr']
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(self.data_dir, dataset_info['extract_dir']), exist_ok=True)
        
        # Download and extract the dataset if needed
        self._download_dataset()
        
        pos_file = os.path.join(self.data_dir, dataset_info['extract_dir'], dataset_info['pos_file'])
        neg_file = os.path.join(self.data_dir, dataset_info['extract_dir'], dataset_info['neg_file'])
        
        # Read positive and negative reviews
        with open(pos_file, 'r', encoding='latin1') as f:
            pos_data = [line.strip() for line in f]
        
        neg_data = []
        with open(neg_file, 'r', encoding='latin1') as f:
            neg_data = [line.strip() for line in f]
        
        # Create DataFrame
        df_pos = pd.DataFrame({'text': pos_data, 'label': 'positive'})
        df_neg = pd.DataFrame({'text': neg_data, 'label': 'negative'})
        
        return pd.concat([df_pos, df_neg], ignore_index=True)
        
    def _load_dataset(self):
        """Load the specified dataset."""
        if self.dataset_name == '20news':
            self.df = self._load_20news()
        elif self.dataset_name == 'mr':
            self.df = self._load_mr()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # Create label map
        self.label_map = {label: i for i, label in enumerate(sorted(self.df['label'].unique()))}
        self.df['label_id'] = self.df['label'].map(self.label_map)
        
        # Preprocess text if needed
        if self.preprocessor:
            self.df['text'] = self.df['text'].apply(self.preprocessor.clean_text)
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\n{'='*50}")
        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"{'='*50}")
        print(f"Total samples: {len(self.df):,}")
        print(f"Number of classes: {len(self.label_map)}")
        print(f"Classes: {', '.join(self.label_map.keys())}")
        
        # Calculate and print average text length
        text_lengths = self.df['text'].apply(lambda x: len(x.split()))
        print(f"Average text length: {text_lengths.mean():.1f} words")
        print(f"Min text length: {text_lengths.min()} words")
        print(f"Max text length: {text_lengths.max()} words")
        
        # Print class distribution
        print("\nClass distribution:")
        class_dist = self.df['label'].value_counts().sort_index()
        for label, count in class_dist.items():
            print(f"- {label}: {count:,} samples ({count/len(self.df)*100:.1f}%)")
        print("="*50 + "\n")
    
    def get_splits(self, return_dataframe=True):
        """
        Split the dataset into train and test sets.
        
        Args:
            return_dataframe (bool): If True, return DataFrames; else return arrays.
            
        Returns:
            tuple: (train_df, test_df) or (X_train, X_test, y_train, y_test)
        """
        # For R8, use the predefined train/test split
        if self.dataset_name == 'r8':
            train_df = self.df.iloc[:5485]  # R8 has 5485 training samples
            test_df = self.df.iloc[5485:]
        else:
            # For MR, use random split
            train_df, test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.df['label_id']
            )
        
        if return_dataframe:
            return train_df, test_df
        else:
            return (
                train_df['text'].values,
                test_df['text'].values,
                train_df['label_id'].values,
                test_df['label_id'].values
            )
    
    def get_num_classes(self):
        """Get the number of classes in the dataset."""
        return len(self.label_map)
    
    def get_label_map(self):
        """Get the mapping from label names to indices."""
        return self.label_map
    
    def get_label_name(self, label_id):
        """Get the label name for a given label ID."""
        id_to_label = {v: k for k, v in self.label_map.items()}
        return id_to_label.get(label_id, 'unknown')
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        class_counts = self.df['label_id'].value_counts().sort_index()
        total = class_counts.sum()
        return {i: total / (len(class_counts) * count) for i, count in class_counts.items()}
