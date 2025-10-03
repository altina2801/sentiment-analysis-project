"""
Text Classification Project - IMDB Sentiment Analysis
Author: AI Internship Challenge
Dataset: IMDB Movie Reviews (50k reviews)
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class SentimentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def load_and_prepare_data(self, file_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:\n{df.head()}")
        
        # Clean the text
        print("\nCleaning text data...")
        df['cleaned_text'] = df['review'].apply(self.clean_text)
        
        return df
    
    def exploratory_analysis(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Class distribution
        print("\nClass Distribution:")
        print(df['sentiment'].value_counts())
        
        # Average review length
        df['review_length'] = df['review'].apply(len)
        df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        
        print(f"\nAverage review length: {df['review_length'].mean():.2f} characters")
        print(f"Average word count: {df['word_count'].mean():.2f} words")
        
        # Most common words
        all_words = ' '.join(df['cleaned_text']).split()
        word_freq = Counter(all_words)
        
        print("\nTop 20 most common words:")
        for word, count in word_freq.most_common(20):
            print(f"{word}: {count}")
        
        # Visualizations
        self.create_visualizations(df, word_freq)
        
    def create_visualizations(self, df, word_freq):
        """Create visualizations for EDA"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        df['sentiment'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].set_xlabel('Sentiment')
        axes[0, 0].set_ylabel('Count')
        
        # Word count distribution
        df.boxplot(column='word_count', by='sentiment', ax=axes[0, 1])
        axes[0, 1].set_title('Word Count by Sentiment')
        axes[0, 1].set_xlabel('Sentiment')
        axes[0, 1].set_ylabel('Word Count')
        
        # Top words
        top_words = dict(word_freq.most_common(15))
        axes[1, 0].barh(list(top_words.keys()), list(top_words.values()))
        axes[1, 0].set_title('Top 15 Most Common Words')
        axes[1, 0].set_xlabel('Frequency')
        
        # Review length distribution
        axes[1, 1].hist([df[df['sentiment']=='positive']['word_count'], 
                        df[df['sentiment']=='negative']['word_count']], 
                       label=['Positive', 'Negative'], bins=50)
        axes[1, 1].set_title('Word Count Distribution')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'eda_visualization.png'")
        plt.close()
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train and compare multiple models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # TF-IDF Vectorization
        print("\nCreating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_tfidf)
            
            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved as 'confusion_matrix_{name.replace(' ', '_')}.png'")
            plt.close()
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.model = results[best_model_name]['model']
        
        print(f"\n{'='*50}")
        print(f"Best Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
        print(f"{'='*50}")
        
        return results
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Vectorize
        vectorized = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.model.predict(vectorized)[0]
        probability = self.model.predict_proba(vectorized)[0]
        
        return {
            'sentiment': prediction,
            'confidence': max(probability),
            'probabilities': {
                'negative': probability[0],
                'positive': probability[1]
            }
        }
    
    def save_model(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Save trained model and vectorizer"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
        """Load trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print("Model and vectorizer loaded successfully!")


def main():
    """Main execution function"""
    # Initialize classifier
    classifier = SentimentClassifier()
    
    # Load data
    # Download dataset from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    # Or use: https://ai.stanford.edu/~amaas/data/sentiment/
    df = classifier.load_and_prepare_data('IMDB Dataset.csv')
    
    # Exploratory analysis
    classifier.exploratory_analysis(df)
    
    # Prepare data for training
    X = df['cleaned_text']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    results = classifier.train_models(X_train, X_test, y_train, y_test)
    
    # Save model
    classifier.save_model()
    
    # Test predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    test_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Do not watch.",
        "It was okay, nothing special but not bad either.",
        "Best movie I've seen this year! Highly recommended!",
        "Boring and predictable. I fell asleep halfway through."
    ]
    
    for review in test_reviews:
        result = classifier.predict(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Negative={result['probabilities']['negative']:.4f}, "
              f"Positive={result['probabilities']['positive']:.4f}")


if __name__ == "__main__":
    main()