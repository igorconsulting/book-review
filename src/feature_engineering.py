import polars as pl
from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)

def add_review_length(df, text_column="text", summary_column="summary"):
    """
    Adds new columns representing the length of each review and summary.
    """
    try:
        df = df.with_columns([
            pl.col(text_column).str.len_chars().alias("review_length"),
            pl.col(summary_column).str.len_chars().alias("summary_length")
        ])
        logger.info("Added review and summary length features successfully.")
    except Exception as e:
        logger.error(f"Error adding review length features: {e}")
        raise e
    return df

def compute_average_scores(df, group_by_col="Title", score_col="score"):
    """
    Computes the average score for each unique item (e.g., each book).
    """
    try:
        avg_scores = df.group_by(group_by_col).agg(pl.col(score_col).mean().alias("avg_score"))
        logger.info("Calculated average score per book successfully.")
        return avg_scores
    except Exception as e:
        logger.error(f"Error in average score calculation: {e}")
        raise e

def compute_review_counts(df, group_by_col="Title"):
    """
    Counts the number of reviews for each unique item (e.g., each book).
    """
    try:
        review_counts = df.group_by(group_by_col).len().alias("review_count")
        logger.info("Calculated review count per book successfully.")
        return review_counts
    except Exception as e:
        logger.error(f"Error in review count calculation: {e}")
        raise e

def add_sentiment_scores(df, sentiment_func, text_column="text"):
    """
    Adds sentiment scores by applying a sentiment analysis function on a text column.
    """
    try:
        sentiment_scores = sentiment_func(df[text_column].to_list())
        df = df.with_columns(pl.Series(sentiment_scores).alias("sentiment_score"))
        logger.info("Added sentiment score successfully.")
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise e
    return df

def add_current_date_column(df):
    """
    Adds a 'current_date' column with today's date in datetime format.
    """
    if "current_date" not in df.columns:
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            df = df.with_columns(
                pl.lit(current_date).str.to_datetime().cast(pl.Date).alias("current_date")
            )
            logger.info("Added 'current_date' column successfully.")
        except Exception as e:
            logger.error(f"Error adding 'current_date' column: {e}")
            raise e
    else:
        logger.info("'current_date' column already exists, skipping step.")
    return df

def convert_time_column_to_date(df):
    """
    Convert the 'time' column to datetime format.
    """
    if "time" in df.columns and df["time"].dtype != pl.Date:
        try:
            df = df.with_columns(
                pl.col("time").str.to_datetime().cast(pl.Date).alias("time")
            )
            logger.info("Converted 'time' column to datetime format successfully.")
        except Exception as e:
            logger.error(f"Error converting 'time' column: {e}")
            raise e
    else:
        logger.info("'time' column is already in datetime format or does not exist, skipping step.")
    return df

def convert_time_columns(df):
    """
    Converts 'time' columns to Date format if they are in string format.
    """
    if "time" in df.columns and df["time"].dtype != pl.Date:
        try:
            df = df.with_columns(
                pl.col("time").str.to_date("%Y-%m-%d").cast(pl.Date).alias("time")
            )
        except Exception as e:
            logger.error(f"Error converting 'time' column: {e}")
            raise e
    return df

def calculate_review_age_days(df):
    """
    Calculate the 'review_age_days' by subtracting 'time' from 'current_date'.
    """
    if "review_age_days" not in df.columns:
        try:
            df = df.with_columns(
                (pl.col("current_date") - pl.col("time")).alias("review_age_days")
            )
            logger.info("Calculated 'review_age_days' successfully.")
        except Exception as e:
            logger.error(f"Error calculating 'review_age_days': {e}")
            raise e
    else:
        logger.info("'review_age_days' column already exists, skipping step.")
    return df

def time_feature_engineering(df):
    """
    Perform feature engineering for date-related attributes.
    """
    logger.info("Starting time-related feature engineering.")
    df = add_current_date_column(df)
    df = convert_time_column_to_date(df)
    df = calculate_review_age_days(df)
    logger.info("Time-related feature engineering completed successfully.")
    return df

def feature_engineering_pipeline(reviews_df, books_info_df, sentiment_func):
    """
    Runs the full feature engineering pipeline on the reviews and books data.
    """
    # Process review lengths and summary lengths
    reviews_df = add_review_length(reviews_df)
    
    # Add sentiment scores
    reviews_df = add_sentiment_scores(reviews_df, sentiment_func)
    
    # Add current date, convert time column, and calculate review age in days
    reviews_df = time_feature_engineering(reviews_df)

    # Compute and join average score and review counts to books_info
    avg_scores = compute_average_scores(reviews_df)
    review_counts = compute_review_counts(reviews_df)
    books_info_df = books_info_df.join(avg_scores, on="Title", how="left")
    books_info_df = books_info_df.join(review_counts, on="Title", how="left")
    
    return reviews_df, books_info_df
