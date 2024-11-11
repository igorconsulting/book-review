import polars as pl

def load_books_data(file_path):
    return pl.read_csv(file_path)

def clean_data(df):
    # Exemplo de limpeza: remover duplicatas e tratar valores ausentes
    df = df.drop_nulls()
    return df