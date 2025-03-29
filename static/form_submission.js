document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("uploadForm");
    const fileInput = document.querySelector('input[type="file"]');
    const loadingScreen = document.getElementById("loadingScreen");
  
    form.addEventListener("submit", function (e) {
      if (!fileInput.value) {
        e.preventDefault();
        alert("📸 Please upload an image before submitting!");
      } else {
        const submitBtn = form.querySelector(".submit-btn");
  
        // Hide form and show loading
        form.classList.add("hidden");
        loadingScreen.classList.remove("hidden");
  
        // Optional: change button text
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
          <span class="inline-block animate-spin mr-2 border-4 border-white border-t-transparent rounded-full w-5 h-5 align-middle"></span>
          🍳 Cooking...
        `;
      }
    });
  });
  