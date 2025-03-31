import pandas as pd
import os
import uuid
import firebase_admin
from firebase_admin import credentials, firestore

class RecipeFirestoreUploader:
    def __init__(self, csv_path: str):
        # Validate CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found {csv_path}")
        
        # Initialize Firebase
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cred_path = os.path.join(base_dir, 'config', 'firebase_config', 'recipegen.json')
        
        try:
            try:
                firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
        
            self.firestore_client = firestore.client()
            self.recipes_collection = self.firestore_client.collection('recipes')
        
        except Exception as e:
            print(f"ERROR Firebase initialization {e}")
            raise
        
        self.csv_path = csv_path
        self.df = None
    
    def load_and_clean_csv(self):
        try:
            self.df = pd.read_csv(self.csv_path, usecols=[
                'Title', 'Ingredients', 'Instructions', 'Image_Name'
            ])
            
            # Missing values replaced with default values
            self.df.fillna('', inplace=True)

            # Convert Ingredients to a list by comma or 
            self.df['Ingredients'] = self.df['Ingredients'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else x.split(", "))
            
            # Convert instructions into a list of steps by new line or period
            self.df['Instructions'] = self.df['Instructions'].apply(lambda x: [step.strip() for step in x.split('.') if step] if isinstance(x, str) else [])

            print(f"Loaded {len(self.df)} unique recipes")
            return self
        
        except Exception as e:
            print(f"ERROR CSV loading {e}")
            raise

    # Upload X at a time to make sure it uploads without batch error
    # 1000 error at 'Loaded 13501 unique recipes'
    # 500 error at 6000
    # 100 error at 600
    def upload_to_firestore(self, batch_size=500):
        if self.df is None:
            print("No data loaded")
            return
        
        try:
            total_recipes = len(self.df)
            
            for start in range(0, total_recipes, batch_size):
                batch = self.firestore_client.batch()
                end = min(start + batch_size, total_recipes)
                
                batch_recipes = self.df.iloc[start:end]
                
                for _, row in batch_recipes.iterrows():
                    existing_recipe = list(self.recipes_collection.where('name', '==', row['Title']).limit(1).stream())
                    if existing_recipe:
                        print(f"Skipping existing recipe: {row['Title']}")
                        continue
                    
                    # Document reference with unique IDs
                    doc_ref = self.recipes_collection.document(str(uuid.uuid4()))
                    
                    # Recipe data structure
                    recipe_data = {
                        'name': row['Title'],
                        'image_name': row['Image_Name'],
                        'ingredients': row['Ingredients'],
                        'instructions': row['Instructions'],
                    }
                    
                    batch.set(doc_ref, recipe_data)
                
                # Commit the batch
                batch.commit()
                print(f"Uploaded recipes {start} to {end}")
            
            print(f"Total recipes uploaded: {total_recipes}")
        
        except Exception as e:
            print(f"ERROR Firestore {e}")
            import traceback
            traceback.print_exc()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_paths = [
        os.path.join(base_dir, 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv'),
        os.path.join(base_dir, 'data', 'Ingredients', 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv'),
        os.path.join(base_dir, 'recipes.csv')
    ]
    
    csv_path = next((path for path in csv_paths if os.path.exists(path)), None)
    
    if not csv_path:
        print("ERROR No CSV found")
        return
    
    try:
        uploader = RecipeFirestoreUploader(csv_path)
        uploader.load_and_clean_csv().upload_to_firestore()
    
    except Exception as e:
        print(f"ERROR {e}")

if __name__ == "__main__":
    main()
