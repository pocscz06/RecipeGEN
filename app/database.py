import os
import pandas as pd
import cv2
import numpy as np
import re
import multiprocessing
import uuid
from typing import List, Tuple
from functools import partial
from sklearn.model_selection import train_test_split
import firebase_admin
from firebase_admin import credentials, firestore, storage

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch

class RecipeProcessorAndUploader:
    def __init__(self, dataset_path: str, csv_path: str, cred_path: str, storage_bucket: str):
        # Validate paths
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"Firebase credentials not found: {cred_path}")
            
        self.dataset_path = dataset_path
        self.csv_path = csv_path
        self.cred_path = cred_path
        self.storage_bucket = storage_bucket
        
        # Initialize data containers
        self.image_paths = []
        self.labels = []
        self.df = None
        self.processed_data = []
        
    
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
    
    def process_recipe_without_firebase(self, row: pd.Series) -> Tuple[str, str]:
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
    
    def load_and_clean_csv(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Select only needed columns if they exist
            columns = ['Title', 'Ingredients', 'Instructions', 'Image_Name']
            existing_columns = [col for col in columns if col in self.df.columns]
            
            # Add missing columns with empty values
            for col in columns:
                if col not in self.df.columns:
                    self.df[col] = ''
            
            # Missing values replaced with default values
            self.df.fillna('', inplace=True)

            # Convert Ingredients to a list
            self.df['Ingredients'] = self.df['Ingredients'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else 
                          x.split(", ") if isinstance(x, str) else []
            )
            
            # Convert instructions into a list of steps
            self.df['Instructions'] = self.df['Instructions'].apply(
                lambda x: [step.strip() for step in x.split('.') if step.strip()] if isinstance(x, str) else []
            )

            print(f"Loaded {len(self.df)} recipes from CSV")
            return self
        
        except Exception as e:
            print(f"ERROR CSV loading: {e}")
            raise
    
    def preprocess_dataset(self, num_workers: int = None) -> None:
        if self.df is None:
            self.load_and_clean_csv()
        
        results = []
        for _, row in self.df.iterrows():
            result = self.process_recipe_without_firebase(row)
            results.append(result)
            
        # Filter out None results
        valid_results = [(img_path, label) for img_path, label in results if img_path is not None]
        
        # Unzip results
        self.image_paths, self.labels = zip(*valid_results) if valid_results else ([], [])
        
        # Update the dataframe with matched images
        image_dict = {self.normalize_name(row['Title']): path 
                      for path, label in zip(self.image_paths, self.labels)
                      for _, row in self.df.iterrows() 
                      if self.normalize_name(row['Title']) == label}
        
        # Add a column for full image paths
        self.df['image_path'] = self.df['Title'].apply(
            lambda title: image_dict.get(self.normalize_name(title), '')
        )
        
        print(f"Total images processed: {len(self.image_paths)}")
        return self
    
    def initialize_firebase(self):
        try:
            try:
                firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(self.cred_path)
                firebase_admin.initialize_app(cred, {
                    "storageBucket": self.storage_bucket
                })
            
            self.firestore_client = firestore.client()
            self.recipes_collection = self.firestore_client.collection('recipes')
            self.storage_bucket = storage.bucket()
            return True
            
        except Exception as e:
            print(f"ERROR Firebase initialization: {e}")
            raise
    
    def upload_to_firebase(self, batch_size=100):
        if self.df is None or 'image_path' not in self.df.columns:
            print("No data loaded or images not processed")
            return self
            
        # Initialize Firebase connections
        if not hasattr(self, 'firestore_client'):
            self.initialize_firebase()
        
        try:
            total_recipes = len(self.df)
            uploaded_count = 0
            
            for start in range(0, total_recipes, batch_size):
                batch = self.firestore_client.batch()
                end = min(start + batch_size, total_recipes)
                
                batch_recipes = self.df.iloc[start:end]
                
                for _, row in batch_recipes.iterrows():
                    # Skip if no image or already exists
                    if not row.get('image_path'):
                        continue
                        
                    # Check if recipe already exists
                    existing_recipe = list(self.recipes_collection
                                          .where('name', '==', row['Title'])
                                          .limit(1)
                                          .stream())
                    if existing_recipe:
                        print(f"Skipping existing recipe: {row['Title']}")
                        continue
                    
                    # Upload image to Firebase Storage
                    image_path = row['image_path']
                    image_filename = os.path.basename(image_path)
                    storage_path = f"recipe_images/{self.normalize_name(row['Title'])}/{image_filename}"
                    
                    blob = self.storage_bucket.blob(storage_path)
                    blob.upload_from_filename(image_path)
                    
                    # Make the blob publicly accessible
                    blob.make_public()
                    
                    # Get the public URL
                    image_url = blob.public_url
                    
                    # Document reference with unique IDs
                    doc_ref = self.recipes_collection.document(str(uuid.uuid4()))
                    
                    # Recipe data structure
                    recipe_data = {
                        'name': row['Title'],
                        'image_name': row.get('Image_Name', ''),
                        'image_url': image_url,
                        'ingredients': row['Ingredients'],
                        'instructions': row['Instructions'],
                    }
                    
                    batch.set(doc_ref, recipe_data)
                    uploaded_count += 1
                
                # Commit the batch
                batch.commit()
                print(f"Uploaded recipes {start} to {end}")
            
            print(f"Total recipes uploaded: {uploaded_count}")
            return self
        
        except Exception as e:
            print(f"ERROR Firebase upload: {e}")
            import traceback
            traceback.print_exc()
            return self
    
    def split_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
        if not self.image_paths:
            raise ValueError("No images processed, preprocess the dataset first")
            
        return train_test_split(
            self.image_paths, 
            self.labels, 
            test_size=test_size, 
            random_state=random_state
        )

def main():
    # Print current working directory and script location for debugging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print(f"Script directory: {script_dir}")
    
    # Use absolute paths to avoid any confusion
    dataset_path = "recipegen/data/ingredients/Food Images"
    csv_path = "recipegen/data/ingredients/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    cred_path = "recipegen/models/recipegen.json"
    
    # Confirm files exist
    print(f"Dataset path exists: {os.path.exists(dataset_path)}")
    print(f"CSV path exists: {os.path.exists(csv_path)}")
    print(f"Credentials path exists: {os.path.exists(cred_path)}")
    
    # Set storage bucket 
    storage_bucket = "recipegen-710d0.firebasestorage.app"
    
    try:
        processor = RecipeProcessorAndUploader(
            dataset_path=dataset_path,
            csv_path=csv_path,
            cred_path=cred_path,
            storage_bucket=storage_bucket
        )
        
        # Chain operations with separate steps
        processor.load_and_clean_csv()
        print("CSV loaded and cleaned successfully")
        
        processor.preprocess_dataset()
        print("Dataset preprocessing completed")
        
        processor.upload_to_firebase()
        print("Firebase upload completed")
        
        # Optionally split the dataset for ML training
        train_images, val_images, train_labels, val_labels = processor.split_dataset()
        print(f"Training set size: {len(train_images)}")
        print(f"Validation set size: {len(val_images)}")
    
    except Exception as e:
        print(f"ERROR in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()