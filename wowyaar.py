import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import os

def preprocess_image(image_path):
    """Preprocess the input image for model prediction."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def load_model():
    """Load the ResNet50 model with custom classification layer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Healthy, Diseased
        model.load_state_dict(torch.load("healthcare_model.pth", map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        raise FileNotFoundError("Model file 'healthcare_model.pth' not found. Ensure it is in the correct directory.")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    return model, device

def predict_disease(image_path, model, device):
    """Predict if the input image is 'Healthy' or 'Diseased'."""
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    
    classes = ['Healthy', 'Diseased']
    return classes[prediction]

if __name__ == "__main__":
    try:
        model, device = load_model()
        test_image_path = "test_medical_scan.jpg"  # Replace with actual image path
        result = predict_disease(test_image_path, model, device)
        print(f"Predicted Diagnosis: {result}")
    except Exception as error:
        print(f"Error: {error}")
