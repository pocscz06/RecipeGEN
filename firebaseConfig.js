// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDrvtmxQ5qGqYPF0v_APGaY1afySo1Q-kI",
  authDomain: "recipegen-710d0.firebaseapp.com",
  projectId: "recipegen-710d0",
  storageBucket: "recipegen-710d0.firebasestorage.app",
  messagingSenderId: "512505420311",
  appId: "1:512505420311:web:8631ca946f41c90c6b1a52",
  measurementId: "G-2KFLTMN6SV",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const analytics = getAnalytics(app);
