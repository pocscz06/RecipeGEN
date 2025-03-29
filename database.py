recipes = []

def add_recipe(name, ingredients, steps):
    recipe = {
        "name": name,
        "ingredients": ingredients,
        "steps": steps
    }
    recipes.append(recipe)

def get_recipes():
    return recipes
