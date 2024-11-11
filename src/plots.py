import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import missingno as msno


# ---------- Missing Data Visualization ----------

def plot_null_matrix(df):
    """
    Plots a null-value matrix to visualize missing data patterns.
    """
    msno.matrix(df)
    plt.title("Null Value Matrix")
    plt.show()

def save_missing_data_heatmap(data, title, save_path):
    """Plot and save a heatmap showing missing data in the dataset."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.is_null().to_pandas(), cbar=False, cmap="viridis")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


# ---------- Distribution Plots ----------

def plot_review_score_distribution(df):
    """
    Plots the distribution of review scores in the dataset.
    """
    sns.countplot(data=df, x='score')
    plt.title("Distribution of Review Scores")
    plt.xlabel("Review Score")
    plt.ylabel("Count")
    plt.show()

def save_histogram(data, column, title, xlabel, ylabel, save_path, bins=20, kde=True):
    """Plot and save a histogram for a specified column."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column].to_numpy(), bins=bins, kde=kde)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

def plot_sentiment_distribution(df, sentiment_column="compound"):
    """
    Plots sentiment scores for reviews, focusing on a specific sentiment column (e.g., compound score).
    """
    sns.histplot(df[sentiment_column], bins=30, kde=True)
    plt.title(f"Distribution of {sentiment_column} Sentiment Scores")
    plt.xlabel(f"{sentiment_column} Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()


# ---------- Scatter and Box Plots ----------

def save_scatterplot(data, x_column, y_column, title, xlabel, ylabel, save_path):
    """Plot and save a scatter plot for two specified columns."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[x_column].to_numpy(), y=data[y_column].to_numpy())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

def save_boxplot(data, column, title, xlabel, save_path):
    """Plot and save a boxplot for a specified column."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[column].to_numpy())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.savefig(save_path)
    plt.close()


# ---------- Word Cloud Visualization ----------

def generate_wordcloud(text, mask=None):
    """
    Generates and displays a word cloud based on input text data.
    """
    wordcloud = WordCloud(mask=mask, background_color='white', colormap='viridis').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    plt.show()


# ---------- Aggregation-Based Bar Plots ----------

def plot_avg_rating_by_category(df):
    """
    Plots a bar chart showing the average rating by book category.
    """
    avg_rating = df.groupby("category").mean("score")
    avg_rating.plot(kind="bar", figsize=(10, 6), color="skyblue")
    plt.title("Average Rating by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Score")
    plt.show()

def plot_avg_rating_by_publisher(df):
    """
    Plots a bar chart showing the average rating by publisher.
    """
    avg_rating = df.groupby("publisher").mean("score")
    avg_rating.plot(kind="bar", figsize=(12, 8), color="lightgreen")
    plt.title("Average Rating by Publisher")
    plt.xlabel("Publisher")
    plt.ylabel("Average Score")
    plt.show()
