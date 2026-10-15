import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image

# Load CLIP Model
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

model, preprocess, tokenizer = load_model()

# Classes
classes = ["persons", "akshay kumar", "salman khan", "boys", "girls", "boy", "girl", "plant", "iphone", "tree", "smartphone", "laptop", "car", "bike", "truck", "bus", "chair", "table", "bottle", "cup", "glass", "pen", "book", "bag", "watch", "shoes", "hat", "headphones", "tv", "remote", "keyboard", "mouse", "mirror", "fan", "clock", "bicycle", "dog", "cat", "bird", "cow", "fish", "frog", "apple", "banana", "orange", "mango", "pineapple", "watermelon", "bread", "pizza", "burger", "ice cream", "chocolate", "cake", "egg", "rice", "tea", "coffee", "spoon", "fork", "plate", "umbrella", "sunglasses", "jacket", "pillow", "bed", "door", "window", "building", "bridge", "road", "train", "ship", "traffic light", "street light", "fire extinguisher", "hand"]

# Encode class labels
@st.cache_resource
def encode_classes():
    text_inputs = tokenizer(classes)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

text_features = encode_classes()

# Streamlit UI
st.title("🖼️ Image Classifier")
st.write("Upload an image and the model will predict what's in it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Classifying..."):
        # Preprocess
        img = image.resize((224, 224))
        img_tensor = preprocess(img).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        pred_class = classes[similarity.argmax().item()]
        confidence = similarity.max().item() * 100
    
    st.success(f"**Prediction:** {pred_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")