import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

# Inicializa o caminho do projeto
project_root = Path("/home/igor/github-projects/book-review")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.paths import *
from src.config import *
import polars as pl

# Carrega o modelo e tokenizador
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print('Foi aqui 1')

# Função de pooling CLS para extrair embeddings
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

# Função para codificar textos
def encode(texts):
    # Tokenizar e computar embeddings dos textos
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)
    embeddings = cls_pooling(model_output)
    return embeddings

# Carrega o dataset model_data
model_data = pl.read_parquet(FEATURE_STORE_DIR / 'model_data.parquet')
print('Foi aqui 2')
# Seleciona uma amostra e combina colunas relevantes em um único texto para cada entrada
sample_data = model_data.sample(fraction=0.0001, seed=42, shuffle=True)
combined_texts = [
    f"Title: {row['Title']}\nDescription: {row['description']}\nAuthors: {row['authors']}\n"
    f"Publisher: {row['publisher']}\nPublished Date: {row['publishedDate']}\nCategories: {row['categories']}\n"
    f"Average Score: {row['avg_score']}\nReview Summary: {row['summary']}\nReview Text: {row['text']}\n"
    for row in sample_data.to_dicts()
]
print('Foi aqui 3')
# Definir perguntas de exemplo
questions = [
    "What is the general opinion about this book?",
    "What do readers think about the book's narrative and style?",
    "Could you summarize the feedback for this book?",
    "What are the strong and weak points mentioned in the reviews?"
]
print('Foi aqui 4')
# Codifica as perguntas e os textos
question_embeddings = encode(questions)
# Função para codificar textos em lotes
def encode_in_batches(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = encode(batch_texts)
        all_embeddings.append(batch_embeddings)
        print(f'Embedou o batch {i}')
    print('Finalizamos o encode_in_batches')
    # Concatena todos os embeddings
    return torch.cat(all_embeddings, dim=0)
print('Foi aqui 5')
context_embeddings = encode_in_batches(combined_texts, batch_size=16)
print('Foi aqui 6')
# Função para encontrar a resposta mais relevante usando pontuação de produto escalar
def get_most_relevant_answer(question_embedding, context_embeddings, contexts):
    scores = torch.mm(question_embedding, context_embeddings.transpose(0, 1))[0].cpu().tolist()
    doc_score_pairs = list(zip(contexts, scores))
    # Ordena pela pontuação em ordem decrescente
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return doc_score_pairs[0]  # Retorna o documento mais relevante

# Testa com uma pergunta de exemplo
sample_question = "What is the best book?"
sample_question_embedding = encode([sample_question])
print('Foi aqui 7')
most_relevant_answer, relevance_score = get_most_relevant_answer(
    sample_question_embedding, context_embeddings, combined_texts
)
print('Foi aqui 8')
print(f"Question: {sample_question}")
print(f"Most Relevant Answer: {most_relevant_answer}")
print(f"Relevance Score: {relevance_score}")