from metaflow import FlowSpec, step
import polars as pl
from extract_rating import convert_time_column, get_unique_books, get_unique_users, get_reviews
from extract_books_info import extract_books_outside_info, extract_books_info
from finetuning import prepare_datasets, fine_tune_model, load_fine_tuned_model
from paths import *
from logger import get_logger
from datetime import datetime
from plots import *

# Initialize the logger
logger = get_logger()

class LLMPipeline(FlowSpec):
    
    @step
    def start(self):
        """Load the raw data."""
        logger.info("Starting the pipeline.")
        
        # Load filtered data files
        try:
            self.model_data = pl.read_csv(FILTERED_DATA_DIR / "model_data.csv")
            logger.info("model data files loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading filtered data: {e}")
            raise e
            
        # Move to fine-tuning step
        self.next(self.fine_tuning)

    @step
    def fine_tuning(self):
        """Fine-tune a transformer model on book review data to improve sentiment analysis."""
        logger.info("Starting fine-tuning on book review data.")

        try:
            reviews = self.books_rating["text"].to_list()
            scores = self.books_rating["score"].to_list()
            
            # Prepare datasets and tokenizer for fine-tuning
            train_dataset, val_dataset, _ = prepare_datasets(reviews, scores)
            
            # Fine-tune the model
            fine_tune_model(train_dataset, val_dataset)
            
            logger.info("Fine-tuning completed and model saved.")
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise e

        # Proceed to prepare_for_modeling step
        self.next(self.prepare_for_modeling)
    
    @step
    def prepare_for_modeling(self):
        """Prepare the data for modeling using the fine-tuned model."""
        logger.info("Preparing data for modeling using the fine-tuned sentiment model.")

        import torch

        # Load fine-tuned model and tokenizer for sentiment analysis
        model, tokenizer = load_fine_tuned_model()

        def transformer_sentiment_analysis(texts):
            """
            Run sentiment analysis using the fine-tuned transformer model.
            """
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.logits.softmax(dim=1).argmax(dim=1).numpy()

        try:
            # Apply transformer-based sentiment analysis on the 'text' column in rating_reviews
            sentiment_scores = transformer_sentiment_analysis(self.books_rating["text"].to_list())
            self.books_rating = self.books_rating.with_columns(
                pl.Series(sentiment_scores).alias("transformer_sentiment_score")
            )
            logger.info("Transformer-based sentiment analysis completed.")

            # Select relevant features for modeling in books_info
            books_info_model = self.books_data.select([
                "Title",
                "description",
                "authors",
                "publisher",
                "publishedDate",
                "categories",
                "book_age",
                "avg_score",
                "review_count"
            ])

            # Select relevant features for modeling in rating_reviews, including transformer sentiment score
            rating_reviews_model = self.books_rating.select([
                "Title",
                "User_id",
                "profileName",
                "score",
                "transformer_sentiment_score",
                "review_length",
                "summary_length",
                "review_age_days"
            ])

            # Save the prepared datasets to the processed directory
            processed_dir = DATA_DIR / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            books_info_model.write_csv(processed_dir / "books_info_for_modeling.csv")
            rating_reviews_model.write_csv(processed_dir / "rating_reviews_for_modeling.csv")
            logger.info("Prepared data saved for modeling in the processed directory.")

        except Exception as e:
            logger.error(f"Error in preparing data for modeling: {e}")
            raise e

        # Move to the next step or finish
        self.next(self.end)

    @step
    def end(self):
        """End of the data pipeline."""
        logger.info("Pipeline completed successfully.")
        print("Pipeline completed successfully.")

if __name__ == "__main__":
    LLMPipeline()
