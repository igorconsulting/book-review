from pathlib import Path

# Define base directories
PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = (PARENT_DIR / "../data").resolve()
PLOTS_DIR = (PARENT_DIR / "../plots").resolve()
# Data subdirectories for different stages of processing
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
FILTERED_DATA_DIR = DATA_DIR / "filtered"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
MODELS_DIR = (PARENT_DIR / "../models").resolve()

# Book and ratings-related constants
BOOKS_DATA_FILE = RAW_DATA_DIR / "books_data.csv"
BOOKS_RATING_FILE = RAW_DATA_DIR / "books_rating.csv"

# Define additional constants as needed
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Link settings for any external data sources (placeholder for additional configurations)
BOOKS_LINK = 'https://example.com/books_data_source'
RATINGS_LINK = 'https://example.com/ratings_data_source'

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, INTERMEDIATE_DATA_DIR, FILTERED_DATA_DIR, MODELS_DIR, FEATURE_STORE_DIR, PLOTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
