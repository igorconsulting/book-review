import polars as pl

def drop_irrelevant_columns(df, columns):
    """
    Drops irrelevant columns from the DataFrame.
    """
    return df.drop(columns)

def fill_null_values(df, text_columns=None, numerical_columns=None, fill_value="Unknown"):
    """
    Fills null values in specified columns.
    - For text columns, fills with a placeholder.
    - For numerical columns, fills with the median value.
    """
    text_columns = text_columns or []
    numerical_columns = numerical_columns or []
    
    # Fill text columns with a placeholder
    for col in text_columns:
        df = df.with_columns(pl.col(col).fill_null(fill_value))
    
    # Fill numerical columns with median values
    for col in numerical_columns:
        median_val = df[col].median()
        df = df.with_columns(pl.col(col).fill_null(median_val))
    
    return df

def remove_duplicates(df, subset):
    """
    Removes duplicate rows based on a subset of columns.
    """
    return df.unique(subset=subset)
