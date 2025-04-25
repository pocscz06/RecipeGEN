from PIL import Image
import os
import numpy as np
from ultralytics import YOLO


def predict_with_yolo(image_path, model_path, conf_threshold=0.25):

    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
   
    #Load model
    model = YOLO(model_path) 
        
    #Ignore confidence level < 25, since lower confidence detections more likely to be unreliable/reduces noise
    model.conf = conf_threshold
        
   
    
    
    #Predictions
    results = model(image_path, verbose=False)
    result = results[0]  
    
    
    #Original image path
    img_filename = os.path.basename(image_path)
    
    #Objection detections
    detections = []
    for i, box in enumerate(result.boxes):
        #Class
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        #Confidence levels
        conf = float(box.conf[0])
        
        #Bounds 
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
        
        #List of all detections
        detections.append({
            'class_name': class_name,
            'confidence': conf,
            'bbox': [x1, y1, x2, y2]
        })
    
    #Predicted image
    output_img_array = result.plot()
    
    #PIL image
    output_img = Image.fromarray(output_img_array[..., ::-1])  
    
    #Save predicted image in predictions directory
    output_dir = os.path.join(os.path.dirname(image_path), 'predictions')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #Save predicted image
    output_path = os.path.join(output_dir, f'pred_{img_filename}')
    output_img.save(output_path)
    
    #Class sort
    class_groups = {}
    for det in detections:
        class_name = det['class_name']
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(det['confidence'])
    
    #Organize
    class_summary = []
    for class_name, confidences in class_groups.items():
        count = len(confidences)
        avg_conf = sum(confidences) / count
        max_conf = max(confidences)
        class_summary.append({
            'class_name': class_name,
            'count': count,
            'max_confidence': max_conf,
            'avg_confidence': avg_conf
        })
    class_summary.sort(key=lambda x: x['count'], reverse=True)
    
    #Return objection detection results!
    return {
        'image': img_filename,
        'detections': detections,
        'detection_count': len(detections),
        'detection_image_path': output_path,
        'class_summary': class_summary
    }