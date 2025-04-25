import os
from dotenv import load_dotenv
load_dotenv() 
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    RESNET_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'ResNet34.pth')
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'best_YOLO_v8.pt')
    print(f"Do files exist? ResNet: {os.path.exists(RESNET_MODEL_PATH)}, YOLO: {os.path.exists(YOLO_MODEL_PATH)}")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  