import polars as pl
from config import DATA_DIR

# Directory path for saving intermediate CSVs
intermediate_dir = DATA_DIR / "intermediate"

def convert_time_column(df):
    """Converts the 'time' column from Unix timestamp (seconds) to datetime format (milliseconds)."""
    return df.with_columns(
        (pl.col("time") * 1_000).cast(pl.Datetime("ms")).alias("time")
    )

def get_unique_books(df):
    """Returns a DataFrame with unique Id and Title values without nulls."""
    unique_books = df.filter(
        pl.col("Id").is_not_null() & pl.col("Title").is_not_null()
    ).select(["Id", "Title"]).unique()
    return unique_books

def get_unique_users(df):
    """Returns a DataFrame with unique User_id and profileName values without nulls."""
    unique_users = df.filter(
        pl.col("User_id").is_not_null() & pl.col("profileName").is_not_null()
    ).select(["User_id", "profileName"]).unique()
    return unique_users

def get_reviews(df):
    """Returns a DataFrame with selected review columns, excluding 'Price'."""
    reviews = df.select(["Title", "User_id", "score", "time", "summary", "text"]).filter(
        pl.col("Title").is_not_null()
    )
    return reviews