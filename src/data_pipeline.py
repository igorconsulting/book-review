from metaflow import FlowSpec, step
import polars as pl
from extract_rating import convert_time_column, get_unique_books, get_unique_users, get_reviews
from extract_books_info import extract_books_outside_info, extract_books_info
from data_cleaning import fill_null_values, remove_duplicates
from feature_engineering import add_review_length, compute_average_scores, add_sentiment_scores
from eda_plots import plot_null_matrix, plot_review_score_distribution, plot_sentiment_distribution, generate_wordcloud
from sentiment_analysis import vader_sentiment_analysis
from config import DATA_DIR, INTERMEDIATE_DATA_DIR

class BookDataPipeline(FlowSpec):
    #1
    @step
    def start(self):
        """Load the raw data."""
        self.books_data = pl.read_csv(DATA_DIR / "raw/books_data.csv")
        self.books_rating = pl.read_csv(DATA_DIR / "raw/books_rating.csv")
        self.next(self.process_books_rating)
    #2
    @step
    def process_books_rating(self):
        """Process books_rating data with conversions and unique extractions."""
        self.books_rating = convert_time_column(self.books_rating)
        self.unique_books = get_unique_books(self.books_rating)
        self.unique_users = get_unique_users(self.books_rating)
        self.reviews = get_reviews(self.books_rating)
        self.next(self.process_books_data)
    #3
    @step
    def process_books_data(self):
        """Process books_data to extract outside info and main book info."""
        self.books_data = self.books_data.drop("ratingsCount")
        self.books_outside_info = extract_books_outside_info(self.books_data)
        self.books_info = extract_books_info(self.books_data)
        self.next(self.clean_data)
    #4
    @step
    def clean_data(self):
        """Clean data by handling null values and removing duplicates."""
        # Drop irrelevant columns and fill missing values
        self.books_info = fill_null_values(self.books_info, text_columns=["description", "authors"])
        self.books_info = remove_duplicates(self.books_info, subset=["Title", "User_id"])
        self.next(self.engineer_features)
    
    @step
    def engineer_features(self):
        """Add engineered features to enrich data for analysis."""
        self.books_info = add_review_length(self.books_info)
        self.books_info = add_sentiment_scores(self.books_info, vader_sentiment_analysis)
        self.average_scores = compute_average_scores(self.books_info, group_by_col="Title", score_col="score")
        self.next(self.perform_eda)
    
    @step
    def perform_eda(self):
        """Perform EDA and visualizations."""
        plot_null_matrix(self.books_info)
        plot_review_score_distribution(self.books_info)
        plot_sentiment_distribution(self.books_info)
        generate_wordcloud(self.books_info["text"].to_list())
        self.next(self.save_data)
    
    @step
    def save_data(self):
        """Save intermediate and enriched data."""
        # Save intermediate and processed outputs
        self.unique_books.write_csv(INTERMEDIATE_DATA_DIR / "rating_unique_books.csv")
        self.unique_users.write_csv(INTERMEDIATE_DATA_DIR / "rating_unique_users.csv")
        self.reviews.write_csv(INTERMEDIATE_DATA_DIR / "rating_reviews.csv")
        self.books_outside_info.write_csv(INTERMEDIATE_DATA_DIR / "books_outside_info.csv")
        self.books_info.write_csv(INTERMEDIATE_DATA_DIR / "books_info.csv")
        self.average_scores.write_csv(INTERMEDIATE_DATA_DIR / "average_scores.csv")
        print("All data saved to intermediate directory.")
        self.next(self.end)
    
    @step
    def end(self):
        """End of the data pipeline."""
        print("Pipeline completed successfully.")

if __name__ == "__main__":
    BookDataPipeline()
