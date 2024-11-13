# Book Reviews Analysis Project

## Project Overview

This project aims to automate the analysis of book reviews, assisting the publishing company by providing insights into book performance, author reputation, genre appeal, and identifying valuable user feedback. The modular approach allows for efficient data cleaning, transformation, sentiment analysis, user segmentation, and advanced insights using machine learning and fine-tuning on large language models (LLMs).

## Project Structure

The project is organized into four main directories and a sequence of scripts and pipeline files that structure the workflow. The components include data processing, feature engineering, model preparation, and fine-tuning for advanced analysis.

### Directory Structure

```
.
├── notebooks/           # Exploratory and analysis notebooks
├── src/                 # Python scripts for processing, feature engineering, and modeling
├── data/                # Folder for storing raw, intermediate, and processed data
└── configs.py           # Global configurations and constants for the project
```

---

### `notebooks/` Directory

The notebooks provide step-by-step processes for loading, cleaning, integrating, analyzing, and visualizing the book review data. 

---

### `src/` Directory

The `src` directory contains core Python scripts and functions for each stage of the analysis pipeline. Each file has a specific role in processing, transforming, or analyzing the data.

#### Key Scripts

1. **`config.py`**  
   - Defines global configurations, including project paths, parameters, and constants for reproducibility and modularity.

2. **`paths.py`**  
   - Manages the directory paths used across scripts, keeping file paths organized and accessible.

3. **`data_processing.py`**  
   - Handles data loading, cleaning, and basic preprocessing. Key functions:
     - **`fill_null_values()`**: Replaces null values in specified columns.
     - **`remove_duplicates()`**: Removes duplicate records based on specified columns.

4. **`data_transformation.py`**  
   - Focuses on data transformation tasks and feature engineering. Key functions:
     - **`add_review_length()`**: Adds columns for review and summary lengths.
     - **`compute_average_scores()`**: Calculates average review scores per book.
     - **`compute_review_counts()`**: Counts reviews per book title.

5. **`sentiment_analysis.py`**  
   - Implements sentiment analysis on review text using sentiment models (VADER, etc.). Key function:
     - **`vader_sentiment_analysis()`**: Analyzes sentiment on review text, returning scores used for user segmentation and insights.

6. **`plots.py`**  
   - Contains reusable plotting functions for visualizations. Key functions:
     - **`plot_review_score_distribution()`**: Plots distribution of review scores.
     - **`plot_sentiment_distribution()`**: Plots sentiment scores across reviews.

7. **`training.py`**  
   - Manages training of machine learning models for tasks like sentiment scoring and recommendation. Key function:
     - **`fine_tune_model()`**: Fine-tunes a pre-trained transformer model to answer book-related questions, based on review data.

8. **`evaluation.py`**  
   - Handles model evaluation metrics and performance tracking.

9. **`pipeline.py`**  
   - Orchestrates the end-to-end data pipeline, defining the sequence of tasks from data ingestion to final model outputs.

---

### Data Pipeline Scripts

These Python scripts define specific workflows, utilizing Metaflow for reproducibility and parallelization.

#### 1. **`data_extraction_pipeline.py`**
   - **Description**: Loads raw data and performs initial extraction tasks, such as getting unique books, users, and preparing reviews for intermediate storage.
   - **Steps**:
     1. **`start`**: Loads raw CSV data.
     2. **`process_books_rating`**: Processes book ratings by extracting unique books, users, and key review details.
     3. **`clean_data`**: Cleans and deduplicates data.
     4. **`save_data`**: Saves intermediate processed data.

#### 2. **`feature_pipeline.py`**
   - **Description**: Generates engineered features from cleaned data, calculating sentiment scores and aggregating review statistics.
   - **Steps**:
     1. **`start`**: Loads filtered data.
     2. **`feature_engineering`**: Applies feature engineering functions to add sentiment scores and calculate review length.
     3. **`save_to_feature_store`**: Saves processed features to the feature store as parquet files for efficient access.

#### 3. **`ml_preparation_pipeline.py`**
   - **Description**: Prepares data for modeling by joining datasets, splitting into train/test sets, and saving for modeling.
   - **Steps**:
     1. **`start`**: Loads data from the feature store.
     2. **`prepare_for_modeling`**: Joins reviews and book metadata, then saves to feature store.
     3. **`split_data`**: Splits the dataset into training and test sets, based on configuration parameters.

#### 4. **`fine-tuning-script.py`**
   - **Description**: Fine-tunes a transformer-based question-answering model on the book review data, enabling it to answer book-specific questions.
   - **Functions**:
     - **`create_combined_text()`**: Prepares a single string for each review by combining all relevant columns.
     - **`fine_tune_model()`**: Fine-tunes a model, saving it in `MODEL_OUTPUT_DIR`.
     - **`ask_model()`**: Tests the model by asking it questions based on sample review data.

#### 5. **`ml_pipeline.py`**
   - **Description**: Fine-tunes and applies an LLM for sentiment scoring and answering contextual questions on book reviews.
   - **Steps**:
     1. **`start`**: Loads and prepares data for fine-tuning.
     2. **`fine_tuning`**: Fine-tunes a transformer model with review-related questions and answers.
     3. **`prepare_for_modeling`**: Prepares data for model analysis using the fine-tuned model.
     4. **`end`**: Concludes the pipeline and saves final model and outputs.

---

## Intermediate Data Warehouse Structure

To efficiently store and access specific subsets of data, we use an intermediate data warehouse structure organized by `data/raw`, `data/intermediate`, and `data/processed` folders. This modular setup optimizes data retrieval and prepares datasets for advanced analysis and modeling.

## Example Workflow

```python
# Example script for loading, processing, and analyzing book review data
import polars as pl
from src.data_processing import fill_null_values, remove_duplicates
from src.sentiment_analysis import vader_sentiment_analysis
from src.feature_engineering import add_review_length

# Load raw data
books_rating = pl.read_csv("data/raw/books_rating.csv")

# Clean data
books_rating = fill_null_values(books_rating)
books_rating = remove_duplicates(books_rating)

# Add features and analyze sentiment
books_rating = add_review_length(books_rating)
books_rating = vader_sentiment_analysis(books_rating["text"])

# Save cleaned and processed data
books_rating.write_csv("data/processed/books_rating_processed.csv")
```

## Fine-Tuning a Transformer Model for Question-Answering on Book Reviews

The `fine-tuning-script.py` is designed to fine-tune a transformer-based question-answering model (e.g., `distilroberta-base`) on the book review data. This enables the model to answer specific questions about books based on the review context, facilitating insights into book summaries, reader feedback, and common themes. This fine-tuning step is integral for adapting a pre-trained language model to the book review domain.

### Script Overview

- **`fine-tuning-script.py`**  
  This script performs data loading, preparation, fine-tuning, and testing on book review data, saving a customized model that can answer book-related questions.

### Steps and Functions in `fine-tuning-script.py`

1. **Load and Sample Data**  
   - `load_data()`: Loads the `model_data.parquet` file from the feature store, containing combined columns of book metadata, review summaries, and reader scores.
   - Samples 10% of the data for training efficiency.
   
2. **Data Preparation**  
   - `create_combined_text(row)`: Generates a comprehensive text block for each book, integrating relevant information such as title, description, authors, categories, and review scores.
   - The model is fine-tuned on this combined information to generate contextually aware answers.

3. **Tokenization**  
   - `tokenize_data(pairs, tokenizer)`: Tokenizes question-context pairs, truncating and padding to ensure consistency in input lengths.
   - Ensures data is formatted correctly for model training.

4. **Fine-Tuning Process**  
   - **Model and Tokenizer Initialization**: Loads the `distilroberta-base` tokenizer and model.
   - **TrainingArguments**: Defines training parameters, including:
     - Batch size: 4
     - Learning rate: `2e-5`
     - Epochs: 3
     - Evaluation strategy: Per epoch
   - **Trainer**: Initializes the Trainer object to manage the fine-tuning loop.
   - **Trainer.train()**: Fine-tunes the model on the training data, optimizing it for question-answering in the book review domain.

5. **Saving the Model**  
   - Saves the fine-tuned model and tokenizer to `MODEL_OUTPUT_DIR` for further use in analysis and querying.

### Sample Questions and Testing

The fine-tuning script includes predefined question templates to test the model’s ability to extract relevant insights from book reviews. Example questions include:

- "What is the general opinion about this book?"
- "What do readers think about the book's narrative and style?"
- "Could you summarize the feedback for this book?"

After training, the model’s responses are tested using the function `ask_model(question, context, model, tokenizer)`, which runs inference on the fine-tuned model and returns contextually relevant answers.

### Usage Example

To run the script:

```bash
python fine-tuning-script.py
```

Upon execution, the script will fine-tune the model, saving the output to the specified directory. Once trained, the model can be used to answer questions about any book in the dataset, providing a powerful tool for querying insights based on real reader feedback.

## Final Remarks

This project setup allows for a modular and scalable approach to data analysis. By organizing data through a staged pipeline, applying feature engineering, and enabling machine learning model training, we provide a comprehensive solution for book review analysis. The intermediate data warehouse and fine-tuning scripts enhance flexibility and ease of access for further analyses, making this setup adaptable for future data additions or modeling tasks.
