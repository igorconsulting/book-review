from metaflow import FlowSpec, step
import polars as pl
from extract_rating import convert_time_column, get_unique_books, get_unique_users, get_reviews
from extract_books_info import extract_books_outside_info, extract_books_info
from paths import *
from logger import get_logger
from datetime import datetime
from plots import *

# Initialize the logger
logger = get_logger()

class BookDataPipeline(FlowSpec):
    
    @step
    def start(self):
        """Load the raw data."""
        logger.info("Starting the pipeline.")
        
        # Load raw data files
        try:
            self.books_data = pl.read_csv(DATA_DIR / "raw/books_data.csv")
            self.books_rating = pl.read_csv(DATA_DIR / "raw/books_rating.csv")
            logger.info("Raw data files loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise e
            
        self.next(self.process_books_rating)
    
    @step
    def process_books_rating(self):
        """Process books_rating data with conversions and unique extractions."""
        logger.info("Processing books_rating data.")
        
        try:
            # Convert time column
            self.books_rating = convert_time_column(self.books_rating)
            logger.info("Time column conversion completed.")
            
            # Get unique books, users, and reviews
            self.unique_books = get_unique_books(self.books_rating)
            self.unique_users = get_unique_users(self.books_rating)
            self.reviews = get_reviews(self.books_rating)
            logger.info("Extracted unique books, users, and reviews.")
        except Exception as e:
            logger.error(f"Error processing books_rating data: {e}")
            raise e
        
        self.next(self.process_books_data)
    
    @step
    def process_books_data(self):
        """Process books_data to extract outside info and main book info."""
        logger.info("Processing books_data to extract main information.")
        
        try:
            # Drop unnecessary columns
            self.books_data = self.books_data.drop("ratingsCount")
            logger.info("Dropped 'ratingsCount' column from books_data.")
            
            # Extract relevant parts
            self.books_outside_info = extract_books_outside_info(self.books_data)
            self.books_info = extract_books_info(self.books_data)
            logger.info("Extracted books_outside_info and books_info.")
        except Exception as e:
            logger.error(f"Error processing books_data: {e}")
            raise e
        
        self.next(self.save_intermediate_data)
    
    @step
    def save_intermediate_data(self):
        """Save intermediate outputs to the data/intermediate directory."""
        logger.info("Saving intermediate outputs to the data/intermediate directory.")
        
        try:
            # Save outputs to CSV
            self.unique_books.write_csv(INTERMEDIATE_DATA_DIR / "rating_unique_books.csv")
            self.unique_users.write_csv(INTERMEDIATE_DATA_DIR / "rating_unique_users.csv")
            self.reviews.write_csv(INTERMEDIATE_DATA_DIR / "rating_reviews.csv")
            self.books_outside_info.write_csv(INTERMEDIATE_DATA_DIR / "books_outside_info.csv")
            self.books_info.write_csv(INTERMEDIATE_DATA_DIR / "books_info.csv")
            logger.info("All data saved to intermediate directory.")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise e
        
        self.next(self.load_intermediate_data)

    @step
    def load_intermediate_data(self):
        """Load intermediate data for further processing."""
        logger.info("Loading intermediate data.")
        
        try:
            # Load intermediate data files
            self.books_info = pl.read_csv(INTERMEDIATE_DATA_DIR / "books_info.csv")
            self.reviews = pl.read_csv(INTERMEDIATE_DATA_DIR / "rating_reviews.csv")
            self.unique_books = pl.read_csv(INTERMEDIATE_DATA_DIR / "rating_unique_books.csv")
            self.unique_users = pl.read_csv(INTERMEDIATE_DATA_DIR / "rating_unique_users.csv")
            self.books_outside_info = pl.read_csv(INTERMEDIATE_DATA_DIR / "books_outside_info.csv")
            
            logger.info("Intermediate data loaded successfully.")
            logger.info(f"books_info shape: {self.books_info.shape}")
            logger.info(f"reviews shape: {self.reviews.shape}")
            logger.info(f"unique_books shape: {self.unique_books.shape}")
            logger.info(f"unique_users shape: {self.unique_users.shape}")
            logger.info(f"books_outside_info shape: {self.books_outside_info.shape}")
        except Exception as e:
            logger.error(f"Error loading intermediate data: {e}")
            raise e
    
        # Move to the next step after loading
        self.next(self.clean_data)


    @step
    def clean_data(self):
        """Clean 'books_info' and 'rating_reviews' tables to prepare for merging."""
        logger.info("Cleaning books_info and rating_reviews data.")

        # Cleaning books_info
        try:
            # Fill missing values in text columns with a placeholder
            self.books_info = self.books_info.with_columns([
                pl.col("description").fill_null("Unknown"),
                pl.col("authors").fill_null("Unknown"),
                pl.col("publisher").fill_null("Unknown"),
                pl.col("categories").fill_null("Uncategorized"),
            ])

            # Drop rows where the Title is missing, as Title is critical for analysis
            self.books_info = self.books_info.filter(pl.col("Title").is_not_null())

            # Log the shape after cleaning
            logger.info(f"books_info cleaned. Shape: {self.books_info.shape}")
        except Exception as e:
            logger.error(f"Error cleaning books_info: {e}")
            raise e

        # Cleaning rating_reviews
        try:
            # Fill missing values in 'summary' and 'text' with placeholder text
            self.reviews = self.reviews.with_columns([
                pl.col("summary").fill_null("No Summary"),
                pl.col("text").fill_null("No Review Text"),
            ])

            # Drop rows where Title or User_id are missing, as these are essential for merging and user analysis
            self.reviews = self.reviews.filter(
                pl.col("Title").is_not_null() & pl.col("User_id").is_not_null()
            )

            # Log the shape after cleaning
            logger.info(f"rating_reviews cleaned. Shape: {self.reviews.shape}")
        except Exception as e:
            logger.error(f"Error cleaning rating_reviews: {e}")
            raise e

        # Move to the next step after cleaning
        self.next(self.save_filtered_data)

    @step
    def save_filtered_data(self):
        """Save cleaned data to the data/filtered directory."""
        logger.info("Saving cleaned data to the data/filtered directory.")


        try:
            # Save cleaned books_info and rating_reviews to CSV
            self.books_info.write_csv(FILTERED_DATA_DIR / "books_info_filtered.csv")
            self.reviews.write_csv(FILTERED_DATA_DIR / "rating_reviews_filtered.csv")

            logger.info("Cleaned data saved successfully to the filtered directory.")
        except Exception as e:
            logger.error(f"Error saving filtered data: {e}")
            raise e

        # Proceed to the next step for feature engineering
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Generate additional features to enrich the data for analysis."""
        logger.info("Starting feature engineering on books_info and rating_reviews.")

        # Feature Engineering for rating_reviews
        try:
            # Review and summary length features
            self.reviews = self.reviews.with_columns([
                pl.col("text").str.len_chars().alias("review_length"),
                pl.col("summary").str.len_chars().alias("summary_length")
            ])

            # Sentiment analysis (using analyze_sentiment function)
            from sentiment_analysis import analyze_sentiment
            sentiment_scores = analyze_sentiment(self.reviews["text"])
            self.reviews = self.reviews.with_columns(
                pl.Series(sentiment_scores).alias("sentiment_score")
            )

            # Calculate review age
            current_date  = datetime.now()
            self.reviews = self.reviews.with_columns(
                (pl.lit(current_date) - pl.col("time")).dt.days().alias("review_age_days")
            )

            logger.info("Feature engineering on rating_reviews completed.")
        except Exception as e:
            logger.error(f"Error during feature engineering on rating_reviews: {e}")
            raise e

        # Feature Engineering for books_info
        try:
            # Book publication recency feature
            current_year = current_date.year
            #current_year = pl.lit(pl.datetime("now").year)
            self.books_info = self.books_info.with_columns(
                (current_year - pl.col("publishedDate").dt.year()).alias("book_age")
            )

            # Average rating per book (joins rating data and calculates average score by Title)
            avg_ratings = self.reviews.groupby("Title").agg(pl.col("score").mean().alias("avg_score"))
            self.books_info = self.books_info.join(avg_ratings, on="Title", how="left")

            # Count of reviews per author (popularity measure)
            author_review_counts = self.reviews.groupby("Title").count().alias("review_count")
            self.books_info = self.books_info.join(author_review_counts, on="Title", how="left")

            logger.info("Feature engineering on books_info completed.")
        except Exception as e:
            logger.error(f"Error during feature engineering on books_info: {e}")
            raise e
        
        # Save engineered features to feature_store
        try:
            self.reviews.write_csv(FEATURE_STORE_DIR / "rating_reviews_features.csv")
            self.books_info.write_csv(FEATURE_STORE_DIR / "books_info_features.csv")
            logger.info("Feature engineered data saved to feature_store successfully.")
        except Exception as e:
            logger.error(f"Error saving feature-engineered data: {e}")
            raise e

        # Proceed to the next step after feature engineering
        self.next(self.perform_eda)

    @step
    def perform_eda(self):
        """Perform an in-depth Exploratory Data Analysis (EDA) on the feature-engineered data."""
        logger.info("Starting EDA on the feature-engineered data.")

        eda_dir = DATA_DIR / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)

        try:

            # 1. Null Value Matrix
            plot_null_matrix(self.books_info)
            logger.info("Null matrix for books_info plotted.")

            # 2. Review Score Distribution
            plot_review_score_distribution(self.reviews)
            logger.info("Review score distribution plotted.")

            # 3. Distribution of Review Sentiments
            plot_sentiment_distribution(self.reviews, sentiment_column="sentiment_score")
            logger.info("Sentiment score distribution plotted.")

            # 4. Distribution of Book Age
            save_histogram(
                self.books_info,
                column="book_age",
                title="Distribution of Book Age",
                xlabel="Book Age (Years)",
                ylabel="Frequency",
                save_path=eda_dir / "book_age_distribution.png"
            )
            logger.info("Book age distribution plot saved.")

            # 5. Distribution of Review Length
            save_histogram(
                self.reviews,
                column="review_length",
                title="Distribution of Review Length",
                xlabel="Review Length (Characters)",
                ylabel="Frequency",
                save_path=eda_dir / "review_length_distribution.png"
            )
            logger.info("Review length distribution plot saved.")

            # 6. Scatter Plot: Book Age vs. Average Score
            save_scatterplot(
                self.books_info,
                x_column="book_age",
                y_column="avg_score",
                title="Book Age vs. Average Score",
                xlabel="Book Age (Years)",
                ylabel="Average Score",
                save_path=eda_dir / "book_age_vs_avg_score.png"
            )
            logger.info("Book Age vs. Average Score scatter plot saved.")

            # 7. Scatter Plot: Review Length vs. Sentiment Score
            save_scatterplot(
                self.reviews,
                x_column="review_length",
                y_column="sentiment_score",
                title="Review Length vs. Sentiment Score",
                xlabel="Review Length (Characters)",
                ylabel="Sentiment Score",
                save_path=eda_dir / "review_length_vs_sentiment_score.png"
            )
            logger.info("Review Length vs. Sentiment Score scatter plot saved.")

            # 8. Missing Data Heatmap for books_info
            save_missing_data_heatmap(
                self.books_info,
                title="Missing Data in Books Info",
                save_path=eda_dir / "books_info_missing_data.png"
            )
            logger.info("Missing data heatmap for books_info saved.")

            # 9. Word Cloud for Book Descriptions
            description_text = " ".join(self.books_info["description"].drop_nulls().to_list())
            generate_wordcloud(description_text)
            logger.info("Word cloud for book descriptions generated.")

        except Exception as e:
            logger.error(f"Error during EDA: {e}")
            raise e

        # Proceed to the next step after EDA
        self.next(self.end)

    @step
    def end(self):
        """End of the data pipeline."""
        logger.info("Pipeline completed successfully.")
        print("Pipeline completed successfully.")

if __name__ == "__main__":
    BookDataPipeline()
