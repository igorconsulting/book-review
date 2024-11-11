def add_feature_avg_rating(df):
    # Exemplo de transformação: calcular média de score por título
    df = df.with_columns((df["score"].mean()).alias("avg_rating"))
    return df