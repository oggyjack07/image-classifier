# 🖼️ AI Image Classifier

An AI-powered image classification web application that identifies and categorizes objects in an uploaded image. Built using OpenAI's CLIP (Contrastive Language-Image Pretraining) model, this app can recognize a wide variety of objects without needing custom model training.

## 🚀 Live Demo
Experience the app live here: **(https://image-classifierr.streamlit.app/)**

## ✨ Features
- **Zero-Shot Classification:** Uses OpenAI's CLIP model to classify images without retraining on a specific dataset.
- **Wide Category Support:** Recognizes 70+ categories including electronics, animals, vehicles, and everyday objects.
- **Real-time Prediction:** Fast and accurate inference directly in the browser.
- **Clean UI:** A user-friendly interface built with Streamlit.
- **Dark/Light Mode:** Seamless theme switching for better user experience.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit (for Web App)
- **AI Model:** OpenAI CLIP (`ViT-B-32`)
- **Libraries:** PyTorch, Open-Clip-Torch, NumPy, Pillow
- **Deployment:** Streamlit Community Cloud

## ⚙️ How It Works
Unlike traditional image classifiers that are trained on a fixed set of images, this project uses **CLIP (Contrastive Language-Image Pretraining)**. 
1. The user uploads an image.
2. The app extracts the image features using a Vision Transformer (ViT).
3. It compares the image features with text embeddings of the provided classes (e.g., "dog", "cat", "laptop").
4. The class with the highest similarity score is displayed as the prediction.

## 🚀 Run Locally

If you want to run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/oggyjack07/your-repo-name.git
