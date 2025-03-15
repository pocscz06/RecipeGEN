# RecipeGEN

Model to scan an image for ingredients and search through a database to fetch potential recipes to be made

## AUTHORS:

KENNY PHAM
HIEU PHUNG
SAMUEL KIM
KEZIAH MUNDLAPATI

### Project Structure:

RecipeGEN/
├── assets/ <!-- Folder for storing external assets (images, icons, etc.)
│   └── images/
│   └── recipes/
├── components/ <!-- JavaScript components for UI elements
├── models/ <!-- Python Machine Learning Models
├── pages/ <!-- HTML pages with component imports
│   └── styles/ <!-- CSS stylesheets
│

This project's general aim is to help users quickly find recipe ideas by submitting an image for review by our trained ML models (CNN--Convolutional Neural Networks, Model #2, Model #3). These models will scan an image of (presumably) ingredients, track which ingredients are in the photo, search through the database for recipes containing those ingredients, and feed them back to the user.

To facilitate this, our front-end is implemented through a web-based application via HTML, CSS, and JavaScript. This front-end utilizes the Tailwind CSS framework for modern styling.

Our back-end is built with Python, using the Flask framework.
As the only purpose of a database within our project is to store kaggle datasets of recipes for our ML models to fetch from and feed back to the user--we find that Firebase--namely Firebase Firestore (NoSQL) is suitable.

#### INSTRUCTIONS TO BUILD AND RUN (For Developers)

<!-- PREREQUISITES -->

Python https://www.python.org/downloads/

Node.js and NPM (Node Package Manager); https://docs.npmjs.com/cli/v11/configuring-npm/install

<!-- DEPENDENCY UPDATES -->

Following convention, our build files/folders are ignored in the Repository via .gitignore to minimize merge conflicts, minimize platform/environment-specific build errors, and repo bloating. As such, whenever you pull changes from the repository that include additions to dependencies within the project's package.json (JS) and requirements.txt (Python) files, you should run:

1. Install NPM packages

```sh
npm install
```

within the terminal to install new dependencies

2. Set up Python Virtual Environment
```sh
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On MacOS/Linux:
source venv/bin/activate
```

3. Install Python dependencies
```sh
pip install -r requirements.txt
```

<!-- RUNNING THE WEB APPLICATION -->

Often, you'll want to test your changes before pushing to the repository. You can do so locally by running the command:

```sh
python app.py
```

This runs our web application locally--only on your system, so that you may view the application and test as needed. After running the command, a development server starts up and makes the application available on a localhost URL that the terminal will provide to you. Simply paste that localhost URL into your designated web browser.
