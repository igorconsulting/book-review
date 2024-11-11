from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from paths import DATA_DIR
import logging

# Initialize logging within finetuning module
logger = logging.getLogger("fine_tuning")

def prepare_datasets(reviews, scores, transformer_model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Tokenize the text data and prepare datasets for training and validation.
    """
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    
    # Tokenize the data
    inputs = tokenizer(reviews, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    # Split data into train and validation sets
    train_idx, val_idx = train_test_split(range(len(reviews)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(inputs["input_ids"][train_idx], torch.tensor(scores)[train_idx])
    val_dataset = torch.utils.data.TensorDataset(inputs["input_ids"][val_idx], torch.tensor(scores)[val_idx])
    
    return train_dataset, val_dataset, tokenizer

def fine_tune_model(train_dataset, val_dataset, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Fine-tune the model on provided datasets and return the fine-tuned model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    training_args = TrainingArguments(
        output_dir=str(DATA_DIR / "models" / "fine_tuned_model"),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=str(DATA_DIR / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning completed.")

    # Save the model and tokenizer
    model.save_pretrained(DATA_DIR / "models" / "fine_tuned_model")

def load_fine_tuned_model(model_path=DATA_DIR / "models" / "fine_tuned_model"):
    """
    Load the fine-tuned model and tokenizer from disk.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer
