import sys
from pathlib import Path

# Define the path to the project's root directory
project_root = Path("/home/igor/github-projects/book-review")

# Add the root directory to sys.path to access the 'src' module
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Define the base data directory using an absolute path
DATA_DIR = project_root / "data"  # Using '/' operator from Path

# Paths to the raw data files
BOOKS_DATA_PATH = DATA_DIR / "raw/books_data.csv"
BOOKS_RATING_PATH = DATA_DIR / "raw/books_rating.csv"

# Path for processed data output
FILTERED_DATA_FILE = DATA_DIR / "processed/books_combined.parquet"

# Other configurations
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_RECOMMENDATIONS = 10

INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"