"""
Simple prediction script for sentiment analysis
Usage: python predict.py
"""

import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
print("Loading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    """Clean and preprocess text data"""
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def load_model():
    """Load trained model and vectorizer"""
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Model loaded successfully!\n")
        return model, vectorizer
    except FileNotFoundError:
        print("✗ Error: Model files not found!")
        print("Please train the model first by running: python main.py")
        return None, None

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for given text"""
    # Clean text
    cleaned = clean_text(text)
    
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    return {
        'sentiment': prediction,
        'confidence': max(probability),
        'probabilities': {
            'negative': probability[0],
            'positive': probability[1]
        }
    }

def print_result(review, result):
    """Pretty print the prediction result"""
    sentiment = result['sentiment']
    confidence = result['confidence'] * 100
    
    emoji = "😊" if sentiment == "positive" else "😞"
    
    print("=" * 60)
    print(f"Review: {review[:100]}{'...' if len(review) > 100 else ''}")
    print("=" * 60)
    print(f"\n{emoji} Sentiment: {sentiment.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nProbabilities:")
    print(f"  Negative: {result['probabilities']['negative']*100:.2f}%")
    print(f"  Positive: {result['probabilities']['positive']*100:.2f}%")
    print("=" * 60 + "\n")

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🎬 MOVIE REVIEW SENTIMENT ANALYZER")
    print("=" * 60 + "\n")
    
    # Load model
    model, vectorizer = load_model()
    if model is None:
        return
    
    # Predefined test reviews
    test_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Complete waste of time and money. Do not watch.",
        "The acting was superb and the story was engaging throughout.",
        "Boring and predictable. I fell asleep halfway through.",
        "Best movie I've seen this year! Highly recommended to everyone!",
        "Worst movie ever. Poor acting, terrible plot, bad directing.",
    ]
    
    while True:
        print("\nOptions:")
        print("1. Test with predefined reviews")
        print("2. Enter your own review")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n" + "=" * 60)
            print("TESTING WITH PREDEFINED REVIEWS")
            print("=" * 60 + "\n")
            
            for i, review in enumerate(test_reviews, 1):
                print(f"\n[Review {i}]")
                result = predict_sentiment(review, model, vectorizer)
                print_result(review, result)
                
        elif choice == "2":
            print("\n" + "=" * 60)
            review = input("Enter your movie review: ").strip()
            
            if not review:
                print("✗ Empty review provided!")
                continue
            
            result = predict_sentiment(review, model, vectorizer)
            print_result(review, result)
            
        elif choice == "3":
            print("\nThank you for using the Sentiment Analyzer! 👋\n")
            break
        else:
            print("✗ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting... 👋\n")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}\n")