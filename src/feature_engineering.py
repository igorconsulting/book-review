import polars as pl

def add_review_length(df, text_column="text"):
    """
    Adds a new column 'review_length' representing the length of each review.
    """
    return df.with_column(pl.col(text_column).str.lengths().alias("review_length"))

def compute_average_scores(df, group_by_col="Title", score_col="score"):
    """
    Computes the average score for each unique item (e.g., each book).
    """
    return df.groupby(group_by_col).agg(pl.col(score_col).mean().alias("average_score"))

def add_sentiment_scores(df, sentiment_func, text_column="text"):
    """
    Adds sentiment scores by applying a sentiment analysis function on a text column.
    The `sentiment_func` should return a dictionary with keys like 'pos', 'neg', 'neu', and 'compound'.
    """
    sentiment_scores = df[text_column].apply(sentiment_func)
    for key in sentiment_scores[0].keys():
        df = df.with_column(pl.Series([s[key] for s in sentiment_scores]).alias(f"sentiment_{key}"))
    return df
