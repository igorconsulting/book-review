from metaflow import FlowSpec, step
import polars as pl
from sklearn.model_selection import train_test_split
from paths import FEATURE_STORE_DIR
from config import RANDOM_SEED, TEST_SIZE  # Importando configurações
from preprocessing import preprocess_data, join_data, train_test_split_df
from logger import get_logger
from utils import save_to_parquet_if_not_exists

# Initialize the logger
logger = get_logger()

class PrepareModelPipeline(FlowSpec):
    
    @step
    def start(self):
        """Load feature-engineered data."""
        logger.info("Loading feature-engineered data from feature_store.")
        
        try:
            # Load data from feature_store
            self.reviews_df = pl.read_parquet(FEATURE_STORE_DIR / "rating_reviews_features.parquet")
            self.books_info_df = pl.read_parquet(FEATURE_STORE_DIR / "books_info_features.parquet")
            logger.info("Feature-engineered data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading feature-engineered data: {e}")
            raise e
        
        self.next(self.prepare_for_modeling)

    @step
    def prepare_for_modeling(self):
        """Apply data preprocessing, join datasets, and prepare for modeling."""
        logger.info("Preparing data for modeling.")

        try:
            # Apply preprocessing pipeline to transform data for modeling
            self.reviews_df, self.books_info_df = preprocess_data(
                reviews=self.reviews_df,
                books_info=self.books_info_df
            )
            logger.info("Data preprocessing completed successfully.")

            # Join the datasets on 'Title' and save the result
            self.model_data = join_data(self.reviews_df, self.books_info_df)
            save_to_parquet_if_not_exists(self.model_data, FEATURE_STORE_DIR / "model_data.parquet", logger)
            logger.info("Joined data saved to feature_store successfully.")

        except Exception as e:
            logger.error(f"Error during data preprocessing or joining: {e}")
            raise e

        self.next(self.split_data)


    @step
    def split_data(self):
        """Split the data into training and test sets using config parameters."""
        logger.info("Splitting data into training and test sets.")
        
        try:
            # Split data
            self.train_data, self.test_data = train_test_split_df(
                self.model_data,
                test_size=TEST_SIZE,
                seed=RANDOM_SEED
            )
            save_to_parquet_if_not_exists(self.train_data, FEATURE_STORE_DIR / "train_data.parquet", logger)
            save_to_parquet_if_not_exists(self.test_data, FEATURE_STORE_DIR / "test_data.parquet", logger)
            logger.info("Data split into training and test sets and saved successfully.")
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise e
        
        self.next(self.end)

    @step
    def end(self):
        """End of the Prepare Model Pipeline."""
        logger.info("Prepare Model Pipeline completed successfully.")
        print("Prepare Model Pipeline completed successfully.")

if __name__ == "__main__":
    PrepareModelPipeline()
