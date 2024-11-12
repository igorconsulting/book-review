from metaflow import FlowSpec, step
import polars as pl
from paths import FILTERED_DATA_DIR, FEATURE_STORE_DIR
from feature_engineering import feature_engineering_pipeline
from logger import get_logger
from sentiment_analysis import analyze_sentiment

# Initialize the logger
logger = get_logger()

class FeaturePipeline(FlowSpec):
    
    @step
    def start(self):
        """Load filtered data."""
        logger.info("Loading filtered data.")
        
        try:
            # Load filtered data files
            self.reviews_df = pl.read_csv(FILTERED_DATA_DIR / "rating_reviews_filtered.csv")
            self.books_info_df = pl.read_csv(FILTERED_DATA_DIR / "books_info_filtered.csv")
            logger.info("Filtered data files loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading filtered data: {e}")
            raise e
        
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Apply feature engineering steps."""
        logger.info("Starting feature engineering on reviews and books data.")
        
        try:
            # Apply feature engineering pipeline
            self.reviews_df, self.books_info_df = feature_engineering_pipeline(
                reviews_df=self.reviews_df,
                books_info_df=self.books_info_df,
                sentiment_func=analyze_sentiment
            )
            logger.info("Feature engineering completed successfully.")
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            raise e
        
        self.next(self.save_to_feature_store)

    @step
    def save_to_feature_store(self):
        """Save engineered features to feature_store."""
        logger.info("Saving feature-engineered data to feature_store.")
        
        try:
            # Define paths for saving
            reviews_path = FEATURE_STORE_DIR / "rating_reviews_features.parquet"
            books_info_path = FEATURE_STORE_DIR / "books_info_features.parquet"
            
            # Save as Parquet files
            self.reviews_df.write_parquet(reviews_path)
            self.books_info_df.write_parquet(books_info_path)
            
            logger.info("Feature-engineered data saved to feature_store successfully.")
        except Exception as e:
            logger.error(f"Error saving feature-engineered data: {e}")
            raise e
        
        self.next(self.end)

    @step
    def end(self):
        """End of the Feature Pipeline."""
        logger.info("Feature Pipeline completed successfully.")
        print("Feature Pipeline completed successfully.")

if __name__ == "__main__":
    FeaturePipeline()
