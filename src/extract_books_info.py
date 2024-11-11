import polars as pl

def extract_books_outside_info(df):
    """
    Extracts columns related to external information about books.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame containing book data.
        
    Returns:
        pl.DataFrame: DataFrame with Title, image, previewLink, and infoLink columns.
    """
    return df.select(["Title", "image", "previewLink", "infoLink"])

def extract_books_info(df):
    """
    Extracts core descriptive columns for books.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame containing book data.
        
    Returns:
        pl.DataFrame: DataFrame with Title, description, authors, publisher, publishedDate, and categories columns.
    """
    return df.select(["Title", "description", "authors", "publisher", "publishedDate", "categories"])
