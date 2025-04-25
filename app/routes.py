import os
from app import app
from flask import Flask, request, redirect, url_for, render_template, send_from_directory,session
from werkzeug.utils import secure_filename
from predict.resnet34_predict import predict_image, ResNet34Classifier
from app.config import Config
from PIL import Image
import logging
import traceback
from app.database import get_recipes_with_ingredient
from predict.yolov8s_predict import predict_with_yolo


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(file):
    if not allowed_file(file.filename):
        return False, "Invalid file type. Only JPG and PNG files are allowed."
    img = Image.open(file)
    img.verify()  
    file.seek(0)
    img = Image.open(file)
    if img.width < 10 or img.height < 10:
        return False, "Image dimensions too small"
    if img.width > 4000 or img.height > 4000:
        return False, "Image dimensions too large"
    file.seek(0)
    return True, None

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predictions/<filename>')
def prediction_file(filename):
    pred_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions')
    return send_from_directory(pred_dir, filename)

#Firebase recipes
@app.route('/recipes/<ingredient>')
def find_recipes(ingredient):
    recipes = get_recipes_with_ingredient(ingredient)
    #Render template
    return render_template(
        'recipe.html',
        title=f"Recipes with {ingredient}",
        ingredient=ingredient,
        recipe_count=len(recipes),
        recipes=recipes
        )


#Logic for handling file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if 'file_to_delete' in session:
        file_path = session.pop('file_to_delete')
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                app.logger.info(f'Deleted file from previous request: {file_path}')
        except Exception as e:
            app.logger.error(f"Failed to delete file: {str(e)}")
    
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
            
        #Validate 
        is_valid = validate_image(file)
        if not is_valid:
            return redirect(request.url)
            
        #Save file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            #Logger for upload
            app.logger.info(f'User uploaded file: {filename}')
            
            #Redirect to the prediction page
            return redirect(url_for('predict_resnet', filename=filename))
    #Render template    
    return render_template('index.html', title="Recipe Generator")


@app.route('/predict/<filename>')
def predict_resnet(filename):
   
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model_path = app.config['RESNET_MODEL_PATH']
        
    logger.info(f"Predicting file: {file_path}")
    logger.info(f"Using model: {model_path}")
        
      
    if not os.path.exists(file_path):
        return redirect(url_for('upload_file'))
            
    if not os.path.exists(model_path):
        return redirect(url_for('upload_file'))
        
    #Prediction function
    try:
            
        logger.info("Successfully imported prediction function")
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return redirect(url_for('upload_file'))
        
    #Prediction (resnet)
    try:
        result = predict_image(file_path, model_path)
        logger.info(f"Prediction result: {result}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return redirect(url_for('upload_file'))
        
        #Render template
    return render_template(
            'resnet.html',
            title="Prediction Result",
            filename=filename,
            result=result
        )



#This route is only for when you click on Detect Objects after performing a ResNet34 prediction!!    
@app.route('/predict_yolo/<filename>')
def predict_yolo(filename):

    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    yolo_model_path = app.config.get('YOLO_MODEL_PATH', 'yolov8s.pt')
        
    logger.info(f"Predicting file with YOLO model: {file_path}")
    logger.info(f"Using YOLO model: {yolo_model_path}")
        
       
    if not os.path.exists(file_path):
        return redirect(url_for('upload_file'))
        
    #YOLO prediction function
    try:
        logger.info("Successfully imported YOLO prediction function")
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return redirect(url_for('upload_file'))
        
    #Prediction (yolo)
    try:
        result = predict_with_yolo(file_path, yolo_model_path)
        logger.info(f"YOLO prediction result: {result}")
            
        #New predictions directory
        pred_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions')
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        #File name of predicted image
        pred_filename = os.path.basename(result['detection_image_path'])
            
    except Exception as e:
        logger.error(f"YOLO prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return redirect(url_for('upload_file'))
        
        #Render template
    return render_template(
            'yolo_v8.html',
            title="YOLO Detection Result",
            filename=filename,
            pred_filename=pred_filename,
            result=result
        )


#This route is only for when you perform the YOLO prediction first!!
@app.route('/predict_direct_yolo', methods=['POST'])
def predict_direct_yolo():
    if 'file' not in request.files:
        return redirect(url_for('upload_file'))
        
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('upload_file'))
        
    #Validate
    is_valid = validate_image(file)
    if not is_valid:
        return redirect(url_for('upload_file'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        app.logger.info(f'User uploaded file for YOLO prediction: {filename}')
        return redirect(url_for('predict_yolo', filename=filename))
        
    return redirect(url_for('upload_file'))
    
