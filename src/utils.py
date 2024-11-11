import polars as pl
import logging
from pathlib import Path

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
