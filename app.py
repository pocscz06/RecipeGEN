from flask import Flask, render_template, request, redirect, url_for
from predict import predict_image
from database import add_recipe, get_recipes
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded'
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict ingredient from image
    predicted_ingredient = predict_image(filepath)

    # Add recipe based on predicted ingredient
    if predicted_ingredient == "carrot":
        add_recipe(
            "Glazed Carrots",
            ["carrots", "brown sugar", "butter"],
            ["Slice carrots", "Sauté with butter", "Add sugar and cook"]
        )
    elif predicted_ingredient == "tomato":
        add_recipe(
            "Tomato Soup",
            ["tomato", "onion", "cream"],
            ["Cook tomatoes", "Blend", "Add cream"]
        )
    elif predicted_ingredient == "onion":
        add_recipe(
            "Onion Rings",
            ["onion", "flour", "egg"],
            ["Slice onion", "Dip & fry"]
        )
    else:
        add_recipe(
            "Mixed Veggie Surprise",
            ["various vegetables"],
            ["Chop everything", "Stir fry with seasoning"]
        )

    recipes = get_recipes()

    return render_template(
        'results.html',
        filename=file.filename,
        ingredients=[predicted_ingredient],
        recipes=recipes
    )

@app.route('/results')
def show_results():
    filename = request.args.get('filename')
    ingredients = ["tomato", "onion", "garlic"]
    recipes = get_recipes()
    return render_template('results.html', filename=filename, recipes=recipes, ingredients=ingredients)

if __name__ == '__main__':
    app.run(debug=True)
