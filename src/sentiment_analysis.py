import polars as pl
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk

# Download the VADER lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize models
sia = SentimentIntensityAnalyzer()
transformer_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
model = AutoModelForSequenceClassification.from_pretrained(transformer_model_name)

def vader_sentiment_analysis(text):
    """
    Performs VADER sentiment analysis on a given text and returns sentiment scores.
    """
    return [sia.polarity_scores(txt)["compound"] for txt in text]

def transformer_sentiment_analysis(text):
    """
    Performs Transformer-based sentiment analysis on a given text using a specified model.
    """
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = softmax(output[0][0].detach().numpy())
    return {'neg': scores[0], 'neu': scores[1], 'pos': scores[2]}

def analyze_sentiment(text, method='vader'):
    """
    Analyzes sentiment of the given text using the specified method.
    """
    if method == 'vader':
        return vader_sentiment_analysis(text)
    elif method == 'transformer':
        return transformer_sentiment_analysis(text)
    else:
        raise ValueError("Invalid sentiment analysis method. Choose from 'vader' or 'transformer'.")