# Challenge


# Notebooks Organization

notebooks/
├── 01_data_loading_and_cleaning.ipynb
├── 02_data_integration.ipynb
├── 03_exploratory_data_analysis.ipynb
├── 04_sentiment_analysis.ipynb
├── 05_user_analysis_and_segmentation.ipynb
├── 06_advanced_insights_and_recommendations.ipynb
└── 07_final_report_and_export.ipynb

# src

src/
├── config.py                # Global configurations and constants for the project
├── paths.py                 # Definition of file and directory paths
├── data_processing.py       # Functions for data loading, cleaning, and preprocessing
├── data_transformation.py   # Functions for data transformation (e.g., feature engineering)
├── plots.py                 # Functions for creating visualizations
├── sentiment_analysis.py    # Functions for sentiment analysis on review text
├── user_analysis.py         # Functions for user analysis and segmentation
├── training.py              # Functions for model training (if machine learning models are involved)
├── evaluation.py            # Functions for evaluating the trained models
├── pipeline.py              # Orchestration of the complete pipeline
└── utils.py                 # Utility functions for the project


# Intermediate Data Warehouse Structure for Book Reviews Project

This document describes the structure and purpose of the intermediate data warehouse used for the Book Reviews project. This setup is designed to organize, clean, and prepare data from `books_rating` for further analysis and modeling, following a modular approach for data transformations and storage.

## Purpose

The intermediate data warehouse serves as a **transitional storage layer** between raw data ingestion and final processing for analysis or machine learning. It focuses on:
- **Data Cleaning**: Handling missing or malformed data.
- **Data Transformation**: Converting Unix timestamps to human-readable dates.
- **Data Segmentation**: Extracting specific subsets of information for individual analyses, such as unique books, unique users, and core review details.

This structure enhances efficiency by breaking down the dataset into organized tables, making it easier to query specific information in later stages.

## Folder Structure

The data warehouse follows a structured folder system within the project, as outlined below:

- **`data/raw/`**: Contains raw, unprocessed files directly from data sources.
  - `books_data.csv`: Metadata about books (e.g., title, author, categories).
  - `books_rating.csv`: User ratings and reviews for books, with Unix timestamps for review dates.

- **`data/intermediate/`**: Stores transformed, intermediate files after basic cleaning and segmentation, serving as a structured repository for quick access to specific datasets.
  - `unique_books.csv`: List of unique book IDs and titles, providing a reference of distinct books.
  - `unique_users.csv`: List of unique user IDs and profile names, capturing distinct users in the dataset.
  - `reviews.csv`: Core review data, including title, user ID, score, time (converted to datetime), summary, and text.

- **`data/processed/`**: Will contain fully processed files ready for analysis or modeling.

## Function Descriptions in `rating_extract.py`

The `rating_extract.py` script provides a set of pure functions to transform and prepare the data. Each function handles a specific transformation, returning a new DataFrame while avoiding side effects, keeping the code modular and testable.

### Core Functions

1. **`convert_time_column(df)`**
   - **Description**: Converts the `time` column from a Unix timestamp in seconds to a datetime format (milliseconds).
   - **Input**: A Polars DataFrame with a `time` column in Unix timestamp format.
   - **Output**: A DataFrame with the `time` column in datetime format.

2. **`get_unique_books(df)`**
   - **Description**: Extracts unique book IDs and titles from the dataset, filtering out null values.
   - **Input**: A Polars DataFrame with `Id` and `Title` columns.
   - **Output**: A DataFrame containing unique book IDs and titles.

3. **`get_unique_users(df)`**
   - **Description**: Extracts unique user IDs and profile names from the dataset, filtering out null values.
   - **Input**: A Polars DataFrame with `User_id` and `profileName` columns.
   - **Output**: A DataFrame with unique user IDs and profile names.

4. **`get_reviews(df)`**
   - **Description**: Creates a DataFrame with core review information (excluding the `Price` column due to high null values).
   - **Input**: A Polars DataFrame with columns `Title`, `User_id`, `score`, `time`, `summary`, and `text`.
   - **Output**: A DataFrame with relevant review details for analysis.

5. **`save_to_csv(df, file_path)`**(located in `utils.py`)
   - **Description**: Saves a given DataFrame to a specified CSV file path in the `data/intermediate` directory.
   - **Input**: A DataFrame and the destination file path.
   - **Output**: Saves the DataFrame as a CSV file.

### Workflow Example

The following script demonstrates the workflow:

```python
import polars as pl
from rating_intermediate import convert_time_column, get_unique_books, get_unique_users, get_reviews, save_to_csv

# Load the raw data
books_rating = pl.read_csv("data/raw/books_rating.csv")

# Apply transformations
books_rating = convert_time_column(books_rating)
unique_books = get_unique_books(books_rating)
unique_users = get_unique_users(books_rating)
reviews = get_reviews(books_rating)

# Save to intermediate folder
save_to_csv(unique_books, "data/intermediate/unique_books.csv")
save_to_csv(unique_users, "data/intermediate/unique_users.csv")
save_to_csv(reviews, "data/intermediate/reviews.csv")
```

### Benefits of this Structure

- **Modular & Testable**: Each function handles a specific task, making code easy to test and modify.
- **Efficiency**: Intermediate tables provide direct access to subsets of data, improving query efficiency and reducing processing time.
- **Organization**: Data is organized in a logical, accessible structure, facilitating further analysis and machine learning tasks.

