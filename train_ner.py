import json
import torch
from transformers import (BertTokenizerFast, BertForTokenClassification,
                          Trainer, TrainingArguments, DataCollatorForTokenClassification)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    """
    Compute metrics for model evaluation.

    Args:
        pred: The predictions from the model.

    Returns:
        dict: A dictionary containing accuracy, F1 score, precision, and recall.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    mask = labels != -100
    labels_filtered = labels[mask]
    preds_filtered = preds[mask]

    precision, recall, f1, _ = precision_recall_fscore_support(labels_filtered,
                                                               preds_filtered,
                                                               average='binary')
    acc = accuracy_score(labels_filtered, preds_filtered)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def prepare_dataset(file_path):
    """
    Prepare the dataset for training by expanding sentences with animal-related questions.

    Args:
        file_path (str): Path to the JSON file containing sentences and labels.

    Returns:
        tuple: A tuple containing expanded sentences and their corresponding labels.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    expanded_sentences = []
    expanded_labels = []

    for sent, label in zip(data['sentences'], data['labels']):
        # Add the original sentence
        expanded_sentences.append(sent)
        expanded_labels.append(label)

        # Find the animal in the sentence
        words = sent.split()
        labels = label
        animal = None
        for word, lab in zip(words, labels):
            if lab == "B-ANIMAL":
                animal = word.lower()
                break

        if animal:
            # Add question forms
            question_templates = [
                f"Is there a {animal} in this image?",
                f"Can you see a {animal} in the picture?",
                f"Does this image contain a {animal}?",
                f"Do you see a {animal}?",
                f"Is that a {animal}?"
            ]

            for template in question_templates:
                words = template.split()
                # Create labels for the new sentence
                new_labels = ["O"] * len(words)
                # Find the index of the animal in the new sentence
                for i, word in enumerate(words):
                    if word.lower() == animal:
                        new_labels[i] = "B-ANIMAL"
                        break

                expanded_sentences.append(template)
                expanded_labels.append(new_labels)

    return expanded_sentences, expanded_labels


# Main training code
sentences, labels = prepare_dataset("data/ner_dataset.json")

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# Tokenization
encodings = tokenizer(sentences,
                      padding=True,
                      truncation=True,
                      return_tensors="pt",
                      return_offsets_mapping=True,
                      return_special_tokens_mask=True)


def align_labels_with_tokens(labels, encodings):
    """
    Align labels with the tokenized input.

    Args:
        labels (list): The original labels for the sentences.
        encodings: The tokenized encodings of the sentences.

    Returns:
        list: A list of aligned labels.
    """
    aligned_labels = []

    for i, label_sequence in enumerate(labels):
        word_ids = encodings.word_ids(i)
        current_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                current_labels.append(-100)
            elif word_id != previous_word_id:
                if word_id < len(label_sequence):
                    current_labels.append(1 if label_sequence[word_id] == "B-ANIMAL" else 0)
                else:
                    current_labels.append(-100)
            else:
                current_labels.append(current_labels[-1] if current_labels else -100)

            previous_word_id = word_id

        aligned_labels.append(current_labels)

    return aligned_labels


aligned_labels = align_labels_with_tokens(labels, encodings)

# Create a dataset from the tokenized inputs
dataset = Dataset.from_dict({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": aligned_labels
})

# Split the dataset into training and testing sets
train_test = dataset.train_test_split(test_size=0.2, seed=42)

# Load the model
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

# Parameter settings for model training
training_args = TrainingArguments(
    output_dir="models/ner_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=10
)

# Create a Trainer object to train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# Start the learning process
trainer.train()

# Saving the trained model and tokenizer
model.save_pretrained("ner_model")
tokenizer.save_pretrained("ner_model")

print("✅ Model training complete and saved to 'ner_model/'")
