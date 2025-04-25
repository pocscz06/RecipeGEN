function displayRecipes(recipesData) {
  const recipesGrid = document.getElementById("recipes-grid");

  if (!recipesData || recipesData.length === 0 || !recipesGrid) {
    console.log("No recipes found or grid element missing");
    return;
  }

  recipesGrid.innerHTML = "";

  const sortedRecipes = [...recipesData].sort(
    (a, b) => (b.ingredient_count || 0) - (a.ingredient_count || 0)
  );

  console.log("Displaying recipes:", sortedRecipes);

  sortedRecipes.forEach((recipe) => {
    const card = document.createElement("div");
    card.className = "recipe-card";
    card.setAttribute("data-recipe-id", recipe.id || "");

    const title = document.createElement("h3");

    title.textContent =
      recipe.recipe_name || recipe.title || recipe.name || "Untitled Recipe";
    card.appendChild(title);

    if (recipe.ingredients && recipe.ingredients.length > 0) {
      const ingredients = document.createElement("div");
      ingredients.className = "recipe-ingredients-preview";

      const topIngredients = recipe.ingredients.slice(0, 3);
      ingredients.textContent = topIngredients.join(", ");

      card.appendChild(ingredients);
    }

    if (recipe.image_url) {
      const img = document.createElement("img");
      img.src = recipe.image_url;
      img.alt = recipe.recipe_name || "Recipe Image";
      img.onerror = function () {
        this.remove();
      };
      card.appendChild(img);
    }

    recipesGrid.appendChild(card);
  });

  console.log(`Displayed ${sortedRecipes.length} recipe cards`);
}

function initRecipeDisplay() {
  const recipeDataElement = document.getElementById("recipe-data");
  let recipeData = [];
  let searchedIngredient = "";

  if (recipeDataElement) {
    try {
      recipeData = JSON.parse(
        recipeDataElement.getAttribute("data-recipes") || "[]"
      );
      searchedIngredient =
        recipeDataElement.getAttribute("data-ingredient") || "";
      console.log(
        `Found data for ${recipeData.length} recipes with ingredient: ${searchedIngredient}`
      );
    } catch (e) {
      console.error("Error parsing recipe data:", e);
    }
  }

  const loadingIndicator = document.querySelector(".loading-indicator");
  if (loadingIndicator) {
    loadingIndicator.remove();
  }

  if (recipeData && recipeData.length > 0) {
    displayRecipes(recipeData);
  } else {
    console.warn("No recipe data found");
    

    const recipesGrid = document.getElementById("recipes-grid");
    if (recipesGrid) {
      recipesGrid.innerHTML = "";
    }
  }
}

document.addEventListener("DOMContentLoaded", initRecipeDisplay);
