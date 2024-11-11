import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import missingno as msno

def plot_null_matrix(df):
    """
    Plots a null-value matrix to visualize missing data patterns.
    """
    msno.matrix(df.to_pandas())
    plt.title("Null Value Matrix")
    plt.show()

def plot_review_score_distribution(df, score_column="score"):
    """
    Plots the distribution of review scores in the dataset.
    """
    sns.countplot(data=df.to_pandas(), x=score_column)
    plt.title("Distribution of Review Scores")
    plt.xlabel("Review Score")
    plt.ylabel("Count")
    plt.show()

def plot_sentiment_distribution(df, sentiment_column="sentiment_compound"):
    """
    Plots the distribution of compound sentiment scores.
    """
    sns.histplot(df[sentiment_column], bins=30, kde=True)
    plt.title("Distribution of Sentiment Scores")
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

def generate_wordcloud(text_data):
    """
    Generates and displays a word cloud from text data.
    """
    wordcloud = WordCloud(background_color='white').generate(" ".join(text_data))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    plt.show()
