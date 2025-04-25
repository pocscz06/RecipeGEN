function displayRecipes(recipesData) {
  const recipesGrid = document.getElementById("recipes-grid");

  if (!recipesData || recipesData.length === 0 || !recipesGrid) {
    console.log("No recipes found or grid element missing");
   
    if (recipesGrid) {
      recipesGrid.innerHTML = "<div class='no-recipes'><p>No recipes were found. Try another ingredient.</p></div>";
    }
    return;
  }


  recipesGrid.innerHTML = "";

  try {
    const sortedRecipes = [...recipesData].sort(
      (a, b) => (b.ingredient_count || 0) - (a.ingredient_count || 0)
    );

    console.log("Displaying recipes:", sortedRecipes);

   
    sortedRecipes.forEach((recipe, index) => {
      console.log(`Processing recipe ${index}:`, recipe);
      
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

      
      const imageUrl = recipe.image_url || recipe.url || recipe.imageUrl;
      console.log(`Recipe ${index} image URL:`, imageUrl);
      
      if (imageUrl) {
        const img = document.createElement("img");
        img.src = imageUrl;
        img.alt = recipe.recipe_name || recipe.name || "Recipe Image";
        img.onerror = function () {
          console.error(`Failed to load image for recipe ${index}:`, imageUrl);
          this.remove();
        };
        img.onload = function() {
          console.log(`Successfully loaded image for recipe ${index}:`, imageUrl);
        };
        card.appendChild(img);
      }

    
      recipesGrid.appendChild(card);
    });

    console.log(`Displayed ${sortedRecipes.length} recipe cards`);
  } catch (error) {
    console.error("Error displaying recipes:", error);
    recipesGrid.innerHTML = "<div class='no-recipes'><p>Error displaying recipes. Please try again.</p></div>";
  }
}


function initRecipeDisplay() {
  console.log("initRecipeDisplay called");
  
 
  const recipeDataElement = document.getElementById("recipe-data");
  console.log("Recipe data element:", recipeDataElement);
  
  let recipeData = [];
  let searchedIngredient = "";

  if (recipeDataElement) {
    const dataAttr = recipeDataElement.getAttribute("data-recipes") || "[]";
    console.log("Raw data-recipes attribute:", dataAttr);
      
    recipeData = JSON.parse(dataAttr);
    console.log("Parsed recipe data:", recipeData);
      
    searchedIngredient =
      recipeDataElement.getAttribute("data-ingredient") || "";
    console.log(
      `Found data for ${recipeData.length} recipes with ingredient: ${searchedIngredient}`
      );
   
  
  } else {
    console.error("Recipe data element not found!");
  }

  const loadingIndicator = document.querySelector(".loading-indicator");
  console.log("Loading indicator:", loadingIndicator);
  
  if (loadingIndicator) {
    loadingIndicator.remove();
    console.log("Removed loading indicator");
  }

  if (recipeData && recipeData.length > 0) {
    console.log("Calling displayRecipes with:", recipeData);
    displayRecipes(recipeData);
  } else {
    console.warn("No recipe data found");

    const recipesGrid = document.getElementById("recipes-grid");
    if (recipesGrid) {
      recipesGrid.innerHTML = "<div class='no-recipes'><p>No recipes were found containing the ingredient. Try another ingredient.</p></div>";
    }
  }
}


document.addEventListener("DOMContentLoaded", function() {
  console.log("Document loaded, initializing recipe display");
  
  setTimeout(initRecipeDisplay, 100);
});
