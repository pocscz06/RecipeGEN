import torch
import torchvision.transforms.v2 as transforms  
from PIL import Image
import os
import numpy as np
from torch import nn
from torchvision import models


#Define NN architecture
class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Classifier, self).__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
#Main predict method
def predict_image(image_path, model_path):
   
   
    print(f"Image path: {image_path}")
    print(f"Model path: {model_path}")
        
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    #Model checkpoint
    required_keys = ['model_state_dict', 'class_mapping'] #Contains model's learning
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: {key}")
        
    #Class mapping
    class_mapping = checkpoint.get('class_mapping', {})
    print(f"Found {len(class_mapping)} classes")
    idx_to_class = {idx: class_name for class_name, idx in class_mapping.items()}
        
    #Load model
    num_classes = len(class_mapping)
    model = ResNet34Classifier(num_classes)
        
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")
    except Exception as e:
        print(f"Error loading model state: {e}")
        raise
            
    model = model.to(device)
    model.eval()  
        
    #ToTensor() is deprecated!!
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and transform image
   
    image = Image.open(image_path).convert('RGB')

    #Batch dimension        
    image_tensor = transform(image).unsqueeze(0)  
    image_tensor = image_tensor.to(device)
        
    #Predictions (calculate confidence levels too!)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    #Class prediction
    class_idx = predicted_class.item()
    class_name = idx_to_class.get(class_idx, f"Unknown (Class {class_idx})")
    confidence_score = confidence.item()
        
    #Top 3 predictions
    top_indices = np.argsort(probabilities.cpu().numpy()[0])[-3:][::-1]
    top_predictions = [(idx_to_class.get(idx, f'Class {idx}'), float(probabilities[0][idx])) 
                           for idx in top_indices]
        
    #class, confidence levels, and top 3 predictions
    result = {
        'class_name': class_name,
        'confidence': confidence_score,
        'top_predictions': top_predictions
    }
        
    print(f"Prediction complete: {class_name} ({confidence_score:.4f})")

    #Return to the frontend!
    return result
        
 


