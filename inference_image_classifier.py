import torch
import json
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import argparse
import os

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Path to the model file
model_path = "image_classifier.pth"

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)

# Load the model using weights
model = models.resnet50(weights='IMAGENET1K_V1')  # Use the appropriate weights
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_path):
    """
    Predict the class of an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the predicted class and confidence score.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]

    # Calculate probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence = probabilities[0][predicted.item()].item()

    return predicted_class, confidence


# CLI for inference
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image.")
    parser.add_argument("image_path", type=str, help="Path to the image")
    args = parser.parse_args()

    result, confidence = predict_image(args.image_path)
    if result is not None:
        print(f"Predicted class: {result} (Confidence: {confidence:.2f})")
