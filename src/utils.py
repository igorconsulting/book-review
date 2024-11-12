import polars as pl
import logging
from pathlib import Path
import os

def save_to_csv(df, file_path):
    """
    Saves a Polars DataFrame to a specified CSV file path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(file_path))
    print(f"Data saved to {file_path}")

def load_csv(file_path):
    """
    Loads a CSV file into a Polars DataFrame.
    """
    return pl.read_csv(file_path)

def configure_logger():
    """
    Configures and returns a logger with colored output.
    """
    logger = logging.getLogger("data_pipeline")
    logger.setLevel(logging.INFO)
    
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

# Function to save DataFrame to Parquet only if the file does not exist
def save_to_parquet_if_not_exists(df, file_path, logger):
    if os.path.exists(file_path):
        logger.info(f"File {file_path} already exists, skipping save.")
    else:
        try:
            df.write_parquet(file_path)
            logger.info(f"File saved successfully to {file_path}.")
        except Exception as e:
            logger.error(f"Error saving file to {file_path}: {e}")
            raise e