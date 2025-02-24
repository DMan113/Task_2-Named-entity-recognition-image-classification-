import torch
import argparse
from transformers import BertTokenizerFast, BertForTokenClassification
import os


def load_model():
    """
    Load the NER model and tokenizer from the specified directory.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    model_path = "models/ner_model"

    # Check if the model directory exists
    if not os.path.exists(model_path):
        print(f"Model directory not found: {model_path}")
        exit(1)

    # Load the model and tokenizer
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer


def predict(text, model, tokenizer):
    """
    Predict named entities in the input text.

    Args:
        text (str): The input text for entity recognition.
        model: The NER model.
        tokenizer: The tokenizer for the model.

    Returns:
        list: A list of detected entities.
    """
    # Encode the input text
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

    with torch.no_grad():
        # Get model outputs
        outputs = model(**{k: v for k, v in encoded.items() if k in ['input_ids', 'attention_mask']})
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert input IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    pred_labels = predictions[0].tolist()

    results = []
    # Extract entities based on predicted labels
    for token, label in zip(tokens, pred_labels):
        if token not in ["[CLS]", "[SEP]"] and label == 1:  # 1 = Animal entity
            results.append(token)

    return results


def main():
    """
    Main function to run the NER inference script.
    """
    parser = argparse.ArgumentParser(description="NER Inference Script")
    parser.add_argument("text", type=str, help="Input text for entity recognition")
    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model()
    # Predict entities in the input text
    entities = predict(args.text, model, tokenizer)

    # Print detected entities
    if entities:
        print("Detected animals:", ", ".join(entities))
    else:
        print("No animals detected.")


if __name__ == "__main__":
    main()
