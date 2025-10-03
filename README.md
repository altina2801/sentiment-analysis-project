# Movie Review Sentiment Analysis

A machine learning project that classifies movie reviews as positive or negative using Natural Language Processing.

## Dataset

**IMDB Movie Reviews Dataset**
- Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Size: 50,000 reviews (25,000 positive, 25,000 negative)
- Format: CSV file with review text and sentiment labels

## Features Implemented

**Core Requirements:**
- Data loading and preprocessing (text cleaning, stopword removal, tokenization)
- Exploratory Data Analysis with visualizations
- Two ML models: Logistic Regression and Naive Bayes
- TF-IDF vectorization with unigrams and bigrams
- 80/20 train-test split with stratification
- Model evaluation (accuracy, precision, recall, F1-score)
- Confusion matrix visualizations
- Prediction interface for new reviews

**Bonus Features:**
- Flask REST API with web interface
- Model persistence (save/load trained models)
- Interactive command-line prediction tool
- Model comparison and performance analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/altina2801/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download dataset:
- Visit the Kaggle link above
- Download IMDB Dataset.csv
- Place it in the project root folder

## Usage

### 1. Train the Model

```bash
python main.py
```

This will:
- Load and clean 50,000 reviews
- Perform exploratory analysis
- Train and compare two models
- Generate visualizations (EDA charts and confusion matrices)
- Save the best model (Logistic Regression, ~89% accuracy)

Training takes approximately 5-10 minutes.

### 2. Test Predictions

**Command-line tool:**
```bash
python predict.py
```
Choose option 1 for predefined examples or option 2 to enter your own review.

**Python script:**
```python
from main import SentimentClassifier

classifier = SentimentClassifier()
classifier.load_model()

result = classifier.predict("This movie was amazing!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 3. Run Web Application

```bash
python app.py
```

Access:
- Web Interface: http://127.0.0.1:5000
- API Endpoint: http://127.0.0.1:5000/api/predict

**API Example:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great movie, loved it!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "probabilities": {
    "negative": 0.08,
    "positive": 0.92
  }
}
```

## Project Structure

```
sentiment-analysis-project/
├── main.py                    # Training pipeline
├── app.py                     # Flask API
├── predict.py                 # Prediction script
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── .gitignore                 # Git ignore rules
├── IMDB Dataset.csv           # Dataset (download required)
├── sentiment_model.pkl        # Trained model (generated)
├── vectorizer.pkl             # TF-IDF vectorizer (generated)
└── *.png                      # Visualizations (generated)
```

## Model Performance

**Logistic Regression (Selected Model):**
- Accuracy: 89.5%
- Precision: 0.89
- Recall: 0.90
- F1-Score: 0.89

**Naive Bayes:**
- Accuracy: 86.5%
- Precision: 0.86
- Recall: 0.87
- F1-Score: 0.86

Logistic Regression was selected for better performance with TF-IDF features and superior handling of feature interactions.

## Technical Details

**Text Preprocessing:**
- Convert to lowercase
- Remove HTML tags and special characters
- Remove stopwords (common English words)
- Tokenize into individual words

**Feature Engineering:**
- TF-IDF vectorization (5,000 features)
- Unigrams and bigrams (1-2 word combinations)
- Captures word importance and context

**Model Training:**
- Stratified 80/20 train-test split
- Maintains class balance in both sets
- Random state 42 for reproducibility

## Dependencies

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
nltk==3.8.1
flask==2.3.3
```

## Troubleshooting

**Model files not found:** Run `python main.py` first to train and save the model.

**NLTK data missing:** The scripts download NLTK data automatically. If issues occur:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**Dataset not found:** Download IMDB Dataset.csv from Kaggle and place it in the project directory.

**Port 5000 in use:** Change port in app.py: `app.run(port=5001)`

## Example Predictions

| Review Text                         | Predicted Sentiment | Confidence |
|-------------------------------------|---------------------|------------|
| Fantastic movie, highly recommend!  | Positive            | 96%        |
| Worst film ever, terrible acting    | Negative            | 94%        |
| Great story and amazing visuals     | Positive            | 93%        |
| Boring and predictable plot         | Negative            | 89%        |



## Author

Altina Hasani
- GitHub: github.com/altina2801
- Email: altinahasani28@gmail.com


