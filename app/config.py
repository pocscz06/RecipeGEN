import os
from dotenv import load_dotenv
load_dotenv() 
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    RESNET_MODEL_PATH = '' #USE CORRECT PATH!!
    YOLO_MODEL_PATH = '' #USE CORRECT PATH!!
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  