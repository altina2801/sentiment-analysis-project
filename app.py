"""
Flask API for Sentiment Classification
Simple REST API to predict sentiment of movie reviews
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# Load model and vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Please train the model first using main.py")
    model = None
    vectorizer = None

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analyzer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 10px;
            display: none;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .positive {
            background: #d4edda;
            border-left: 5px solid #28a745;
        }
        .negative {
            background: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .result h2 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .confidence {
            font-size: 1.2em;
            margin-top: 10px;
        }
        .probabilities {
            margin-top: 15px;
            display: flex;
            gap: 20px;
        }
        .prob-item {
            flex: 1;
            padding: 10px;
            background: rgba(255,255,255,0.5);
            border-radius: 5px;
            text-align: center;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
            color: #667eea;
            font-weight: bold;
        }
        .emoji {
            font-size: 3em;
            margin-bottom: 10px;
        }
        .examples {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
        }
        .examples h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .example-btn {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        .example-btn:hover {
            background: #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Movie Review Analyzer</h1>
        <p class="subtitle">Analyze the sentiment of any movie review</p>
        
        <form id="sentimentForm">
            <textarea id="reviewText" placeholder="Enter a movie review here..." required></textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>
        
        <div class="loading" id="loading">Analyzing...</div>
        
        <div id="result" class="result"></div>
        
        <div class="examples">
            <h3>Try these examples:</h3>
            <div class="example-btn" onclick="fillExample('This movie was absolutely fantastic! I loved every minute of it.')">Positive Review</div>
            <div class="example-btn" onclick="fillExample('Terrible film. Waste of time and money. Do not watch.')">Negative Review</div>
            <div class="example-btn" onclick="fillExample('The cinematography was stunning, but the plot was confusing.')">Mixed Review</div>
        </div>
    </div>

    <script>
        function fillExample(text) {
            document.getElementById('reviewText').value = text;
        }

        document.getElementById('sentimentForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = document.getElementById('reviewText').value;
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            // Show loading
            loadingDiv.style.display = 'block';
            resultDiv.style.display = 'none';
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Hide loading
                loadingDiv.style.display = 'none';
                
                // Display result
                const sentiment = data.sentiment;
                const isPositive = sentiment === 'positive';
                
                resultDiv.className = `result ${sentiment}`;
                resultDiv.innerHTML = `
                    <div class="emoji">${isPositive ? '😊' : '😞'}</div>
                    <h2>Sentiment: ${sentiment.toUpperCase()}</h2>
                    <div class="confidence">
                        Confidence: ${(data.confidence * 100).toFixed(2)}%
                    </div>
                    <div class="probabilities">
                        <div class="prob-item">
                            <strong>Negative</strong><br>
                            ${(data.probabilities.negative * 100).toFixed(2)}%
                        </div>
                        <div class="prob-item">
                            <strong>Positive</strong><br>
                            ${(data.probabilities.positive * 100).toFixed(2)}%
                        </div>
                    </div>
                `;
                resultDiv.style.display = 'block';
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultDiv.className = 'result negative';
                resultDiv.innerHTML = `<h2>Error</h2><p>${error.message}</p>`;
                resultDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for sentiment prediction"""
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.'
            }), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({
                'error': 'Empty text provided.'
            }), 400
        
        # Clean and vectorize
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        return jsonify({
            'sentiment': prediction,
            'confidence': float(max(probability)),
            'probabilities': {
                'negative': float(probability[0]),
                'positive': float(probability[1])
            },
            'original_text': text
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Sentiment Analysis API Server")
    print("="*50)
    print("\nStarting Flask server...")
    print("Web Interface: http://127.0.0.1:5000")
    print("API Endpoint: http://127.0.0.1:5000/api/predict")
    print("\nAPI Usage Example:")
    print('curl -X POST http://127.0.0.1:5000/api/predict \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"text": "This movie was great!"}\'')
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)