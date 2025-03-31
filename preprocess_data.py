import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import re
import multiprocessing
from functools import partial

class ImageDatasetPreprocessor:
    def __init__(self, dataset_path: str, csv_path: str):
        self.dataset_path = dataset_path
        self.csv_path = csv_path
        self.image_paths = []
        self.labels = []
    

    def normalize_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        name = name.lower()
        # Remove special characters
        name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with dashes
        name = re.sub(r'\s+', '-', name)
        return name
    

    def find_matching_image(self, recipe_name: str) -> str:
        normalized_recipe = self.normalize_name(recipe_name)
        for filename in os.listdir(self.dataset_path):
            # Normalize image file name
            normalized_filename = self.normalize_name(os.path.splitext(filename)[0])
            # Flexible matching
            if normalized_recipe in normalized_filename or normalized_filename in normalized_recipe:
                return os.path.join(self.dataset_path, filename)
        
        return None
    
    # Process each recipe to find matching images and normalize names
    def process_recipe(self, row: pd.Series) -> Tuple[str, str]:
        recipe_name = row['Title']
        img_path = self.find_matching_image(recipe_name)
        
        if not img_path or not os.path.exists(img_path):
            return None, None
        
        try:
            # Reading image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is None or img.size == 0:
                return None, None
            
            return img_path, self.normalize_name(recipe_name)
        
        except Exception as e:
            print(f"ERROR {img_path}: {e}")
            return None, None
    
    def preprocess_dataset(self, num_workers: int = None) -> None:
        # Load CSV
        df = pd.read_csv(self.csv_path)
        # Multiprocessing
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(self.process_recipe, [row for _, row in df.iterrows()])
        # Filter out None
        valid_results = [(img_path, label) for img_path, label in results if img_path is not None]
        # Results
        self.image_paths, self.labels = zip(*valid_results) if valid_results else ([], [])
        print(f"Total images processed: {len(self.image_paths)}")
    
    # Dataset splits into training and validation sets
    def split_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
        if not self.image_paths:
            raise ValueError("No images processed, preprocess again")
        return train_test_split(
            self.image_paths, 
            self.labels, 
            test_size=test_size, 
            random_state=random_state
        )

# Main
def main():
    DATASET_PATH = "data/ingredients/Food Images"
    CSV_PATH = "data/ingredients/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    # Initialize preprocessor
    preprocessor = ImageDatasetPreprocessor(DATASET_PATH, CSV_PATH)
    # Preprocess dataset
    preprocessor.preprocess_dataset()
    # Split dataset
    train_images, val_images, train_labels, val_labels = preprocessor.split_dataset()
    print(f"Training set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")

if __name__ == "__main__":
    main()