import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn


class AnimalClassificationPipeline:
    def __init__(self, ner_model_path="models/ner_model", image_model_path="models/image_classifier.pth"):
        """
        Initialize the Animal Classification Pipeline.

        Args:
            ner_model_path (str): Path to the NER model directory.
            image_model_path (str): Path to the image classification model file.
        """
        # Load NER model
        self.ner_tokenizer = BertTokenizerFast.from_pretrained(ner_model_path)
        self.ner_model = BertForTokenClassification.from_pretrained(ner_model_path)
        self.ner_model.eval()

        # Load image classification model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_model = models.resnet50(weights=None)  # Initially without weights
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 10)
        self.image_model.load_state_dict(torch.load(image_model_path))
        self.image_model.to(self.device)
        self.image_model.eval()

        # Animal classes
        self.animal_classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
                               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

        # Image transformations
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_animal_from_text(self, text):
        """
        Extract animal names from the input text using the NER model.

        Args:
            text (str): The input text for entity recognition.

        Returns:
            str: The extracted animal name or None if no animal is found.
        """
        encoded = self.ner_tokenizer(text,
                                     return_tensors="pt",
                                     return_offsets_mapping=True,
                                     return_special_tokens_mask=True)

        with torch.no_grad():
            outputs = self.ner_model(**{k: v for k, v in encoded.items()
                                        if k in ['input_ids', 'attention_mask']})
            predictions = torch.argmax(outputs.logits, dim=-1)

        tokens = self.ner_tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        pred_labels = predictions[0].tolist()

        animal_tokens = []
        for token, label in zip(tokens, pred_labels):
            if label == 1:  # 1 indicates an animal entity
                if token.startswith("##"):
                    animal_tokens.append(token[2:])
                else:
                    animal_tokens.append(token)

        if animal_tokens:
            return "".join(animal_tokens).lower()
        return None

    def classify_image(self, image_path):
        """
        Classify the animal in the given image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The predicted animal class.
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.image_model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        return self.animal_classes[predicted.item()]

    def process(self, text, image_path):
        """
        Process the input text and image to determine if they match.

        Args:
            text (str): The input text for entity recognition.
            image_path (str): The path to the image file.

        Returns:
            tuple: A tuple containing a boolean indicating if there is a match and a dictionary with details.
        """
        text_animal = self.extract_animal_from_text(text)
        if not text_animal:
            return False, "No animal mentioned in text"

        image_animal = self.classify_image(image_path)

        return text_animal == image_animal, {
            'text_animal': text_animal,
            'image_animal': image_animal
        }


if __name__ == "__main__":
    pipeline = AnimalClassificationPipeline()

    text = "Is there a spider in this image?"
    image_path = "data/processed/test/spider/OIP-0Ivnzv3__K8m_klHuBVFpAHaFi.jpeg"

    result, details = pipeline.process(text, image_path)
    print(f"Match: {result}")
    print(f"Details: {details}")
