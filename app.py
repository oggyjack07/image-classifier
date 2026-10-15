import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image

# ---------- Page Config ----------
st.set_page_config(
    page_title="Image Classifier | Suraj Verma",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Hide Streamlit default header/footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }
    
    /* Title */
    .title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f1f5f9;
        text-align: center;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Card */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35);
    }
    
    .result-label {
        color: #bfdbfe;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .result-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        text-transform: capitalize;
        margin-bottom: 0.3rem;
    }
    
    .result-confidence {
        color: #dbeafe;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #0f172a;
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #1e293b;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Image */
    [data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Divider */
    hr {
        border-color: #334155;
        margin: 2rem 0;
    }
    
    /* Footer text */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource(show_spinner=False)
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

# ---------- Classes ----------
classes = ["persons", "akshay kumar", "salman khan", "boys", "girls", "boy", "girl", "plant", "iphone", "tree", "smartphone", "laptop", "car", "bike", "truck", "bus", "chair", "table", "bottle", "cup", "glass", "pen", "book", "bag", "watch", "shoes", "hat", "headphones", "tv", "remote", "keyboard", "mouse", "mirror", "fan", "clock", "bicycle", "dog", "cat", "bird", "cow", "fish", "frog", "apple", "banana", "orange", "mango", "pineapple", "watermelon", "bread", "pizza", "burger", "ice cream", "chocolate", "cake", "egg", "rice", "tea", "coffee", "spoon", "fork", "plate", "umbrella", "sunglasses", "jacket", "pillow", "bed", "door", "window", "building", "bridge", "road", "train", "ship", "traffic light", "street light", "fire extinguisher", "hand"]

@st.cache_resource(show_spinner=False)
def encode_classes():
    model, preprocess, tokenizer = load_model()
    text_inputs = tokenizer(classes)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

# ---------- UI ----------
st.markdown('<div class="title">🖼️ Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let AI identify what\'s in it</div>', unsafe_allow_html=True)

# Load model with spinner
with st.spinner("Loading model..."):
    model, preprocess, tokenizer = load_model()
    text_features = encode_classes()

# Card with uploader
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("**Uploaded Image**")
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("**Prediction**")
        with st.spinner("Classifying..."):
            img = image.resize((224, 224))
            img_tensor = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            pred_class = classes[similarity.argmax().item()]
            confidence = similarity.max().item() * 100
        
        st.markdown(f'''
        <div class="result-box">
            <div class="result-label">Detected</div>
            <div class="result-value">{pred_class}</div>
            <div class="result-confidence">Confidence: {confidence:.2f}%</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<div class="footer-text">Built by Suraj Verma • Powered by CLIP</div>', unsafe_allow_html=True)
