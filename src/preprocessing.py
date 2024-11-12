import polars as pl
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def fill_nulls(df, columns_to_fill=None, fill_value="Unknown"):
    """
    Fills null values in specified columns with a given fill_value.
    """
    if columns_to_fill:
        for col in columns_to_fill:
            df = df.with_columns(pl.col(col).fill_null(fill_value))
    return df

def convert_time_columns(df, date_column="time", current_date_column="current_date"):
    """
    Converts specified time columns to datetime.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    df = df.with_columns([
        pl.lit(current_date).str.to_datetime().cast(pl.Date).alias(current_date_column),
        pl.col(date_column).str.to_datetime().cast(pl.Date)
    ])
    return df

def apply_text_length_features(df, text_columns=None):
    """
    Adds text length features for specified text columns.
    """
    if text_columns:
        for text_col in text_columns:
            df = df.with_columns(pl.col(text_col).str.len_chars().alias(f"{text_col}_length"))
    return df

def preprocess_data(reviews, books_info):
    """
    Preprocess both reviews and books_info DataFrames with all preprocessing steps.
    """
    logger.info("Starting preprocessing on reviews and books_info")

    # Exclude the 'time' column
    reviews = reviews.drop("time")
    # Fill null values
    reviews = fill_nulls(reviews, columns_to_fill=["summary", "text", "User_id"])
    books_info = fill_nulls(books_info, columns_to_fill=["description", "authors", "publisher", "categories"])

    # Convert time columns
    #reviews = convert_time_columns(reviews)

    # Apply text length features
    reviews = apply_text_length_features(reviews, text_columns=["text", "summary"])

    logger.info("Preprocessing completed.")
    return reviews, books_info

def join_data(reviews_df, books_info_df, join_column="Title"):
    """
    Joins the reviews and books_info DataFrames on a specified column.
    
    Parameters:
    - reviews_df (pl.DataFrame): DataFrame containing reviews data.
    - books_info_df (pl.DataFrame): DataFrame containing books info data.
    - join_column (str): Column name to join on, default is 'Title'.
    
    Returns:
    - pl.DataFrame: Joined DataFrame.
    """
    try:
        joined_df = reviews_df.join(books_info_df, on=join_column, how="left")
        logger.info("Data joined successfully on column: %s", join_column)
        return joined_df
    except Exception as e:
        logger.error(f"Error during data join on column '{join_column}': {e}")
        raise e
    
def train_test_split_df(df, seed=0, test_size=0.2):
    return df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .shuffle(seed=seed)
        .gt(pl.len() * test_size)
        .alias("split")
    ).partition_by("split", include_key=False)
