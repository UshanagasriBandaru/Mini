import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

# Generate synthetic neutral examples
def generate_neutral_examples(n_samples=2000):
    mental_health_questions = [
        "What are effective ways to manage stress?",
        "How can I practice mindfulness every day?",
        "What is cognitive behavioral therapy and how does it work?",
        "How do I cope with social anxiety?",
        "What are the mental health benefits of meditation?",
        "How can I improve my sleep for better mental health?",
        "What’s the difference between stress and anxiety?",
        "How do I set healthy boundaries in relationships?",
        "What are grounding techniques for panic attacks?",
        "How does journaling support emotional wellness?",
        "What is self-care and why is it important?",
        "How can I deal with feelings of overwhelm?",
        "What are signs of depression to watch for?",
        "How does exercise impact mental health?",
        "What is the role of a therapist in mental health care?",
        "How can I build resilience to stress?",
        "What are relaxation techniques for anxiety?",
        "How do I practice gratitude for mental wellness?",
        "What is the 5-4-3-2-1 grounding technique?",
        "How can I support a friend with mental health challenges?",
        "What are coping strategies for low mood?",
        "How does diet affect mental health?",
        "What is mindfulness-based stress reduction?",
        "How can I reduce negative self-talk?",
        "What are the benefits of deep breathing exercises?",
        "How do I create a mental health routine?",
        "What is exposure therapy for anxiety?",
        "How can I manage work-related stress?",
        "What are the effects of social media on mental health?",
        "How do I find a mental health professional?"
    ]
    neutral_texts = []
    for i in range(n_samples):
        base_text = mental_health_questions[i % len(mental_health_questions)]
        if i % 2 == 0:
            neutral_texts.append(base_text.replace("?", f" {i}?"))
        else:
            neutral_texts.append(f"{base_text} (example {i})")
    return neutral_texts

# Load and augment dataset
def prepare_dataset():
    dataset = load_dataset("emotion")
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    neutral_train_texts = generate_neutral_examples(n_samples=1600)
    neutral_val_texts = generate_neutral_examples(n_samples=200)

    neutral_train_df = pd.DataFrame({"text": neutral_train_texts, "label": 6})
    neutral_val_df = pd.DataFrame({"text": neutral_val_texts, "label": 6})

    train_df = pd.concat([train_df, neutral_train_df], ignore_index=True)
    val_df = pd.concat([val_df, neutral_val_df], ignore_index=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Main training function
def train_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset, val_dataset = prepare_dataset()

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=7)

    training_args = TrainingArguments(
        output_dir="./model",
        eval_strategy="epoch",  # Change 'eval uation_strategy' to 'eval_strategy'
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()

    train_losses = []
    eval_losses = []
    log_history = trainer.state.log_history
    for log in log_history:
        if "loss" in log and "step" in log:
            train_losses.append(log["loss"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])

    epochs = range(1, len(eval_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses[:len(epochs)], label="Training Loss")
    plt.plot(epochs, eval_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig("loss_vs_epoch.png")
    print("Loss vs Epoch graph saved as 'loss_vs_epoch.png'")
    plt.close()

    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    print("Model and tokenizer saved to ./model")

    eval_metrics = trainer.evaluate()
    print(f"Final accuracy: {eval_metrics['eval_accuracy']:.4f}")
    print(f"Final F1-score: {eval_metrics['eval_f1']:.4f}")

if __name__ == "__main__":
    os.makedirs("./model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    train_model()
