import sys
from pathlib import Path
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from paths import FEATURE_STORE_DIR
from config import *
import json

# Definir o caminho do diretório raiz do projeto
project_root = Path("/home/igor/github-projects/book-review")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Caminho para salvar o modelo fine-tunado
MODEL_OUTPUT_DIR = Path("./fine_tuned_model")

# Verificação de GPU disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função de carregamento dos dados
def load_data():
    model_data = pl.read_parquet(FEATURE_STORE_DIR / "model_data.parquet")
    return model_data.sample(fraction=0.1, seed=42, shuffle=True)

# Função para combinar as informações em uma string única para cada livro
def create_combined_text(row):
    return (
        f"Title: {row['Title']}\n"
        f"Description: {row['description']}\n"
        f"Authors: {row['authors']}\n"
        f"Publisher: {row['publisher']}\n"
        f"Published Date: {row['publishedDate']}\n"
        f"Categories: {row['categories']}\n"
        f"Average Score: {row['avg_score']}\n"
        f"Review Summary: {row['summary']}\n"
        f"Review Text: {row['text']}\n"
        f"Review Score: {row['score']}\n"
    )

# Função para tokenizar pares de perguntas e contextos
def tokenize_data(pairs, tokenizer):
    inputs = tokenizer(
        [pair["question"] for pair in pairs],
        [pair["context"] for pair in pairs],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    return inputs

# Função principal para o fine-tuning do modelo
def fine_tune_model():
    # Carregar dados e gerar textos combinados
    model_data = load_data()
    combined_texts = [create_combined_text(row) for row in model_data.to_dicts()]
    
    # Perguntas para treinamento
    questions = [
        "What is the general opinion about this book?",
        "What do readers think about the book's narrative and style?",
        "Could you summarize the feedback for this book?",
        "What are the strong and weak points mentioned in the reviews?"
    ]
    
    # Preparar pares de treinamento
    training_pairs = [{"question": q, "context": combined_text} for combined_text in combined_texts for q in questions]

    # Inicializar o tokenizer e o modelo para fine-tuning
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Tokenizar dados
    inputs = tokenize_data(training_pairs, tokenizer)
    
    # Argumentos para o treinamento
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="none"
    )

    # Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs
    )

    # Fine-tune o modelo
    trainer.train()

    # Salvar modelo fine-tunado e tokenizer
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Fine-tuned model saved to {MODEL_OUTPUT_DIR}")

# Função para testar o modelo com uma amostra de pergunta e contexto
def ask_model(question, context, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores, end_scores = outputs.start_logits, outputs.end_logits
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx]))
    return answer

# Função principal para execução completa
def main():
    print("device:", device)
    fine_tune_model()

    # Testar o modelo fine-tunado
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)

    # Testar com uma pergunta e contexto de exemplo
    sample_question = "What is the general opinion about this book?"
    sample_context = create_combined_text(model_data[0].to_dicts()[0])  # Usando a primeira amostra
    print("Sample Question:", sample_question)
    print("Model's Answer:", ask_model(sample_question, sample_context, model, tokenizer))

if __name__ == "__main__":
    main()
