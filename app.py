import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image

# ---------- Page Config ----------
st.set_page_config(
    page_title="Image Classifier | Suraj Verma",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%233b82f6'><path d='M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z'/></svg>",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }
    
    /* Title with icon */
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        margin-bottom: 0.5rem;
    }
    
    .header svg {
        width: 44px;
        height: 44px;
        stroke: #3b82f6;
        stroke-width: 2;
        fill: none;
    }
    
    .title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f1f5f9;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Section label with icon */
    .section-label {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #cbd5e1;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-label svg {
        width: 18px;
        height: 18px;
        stroke: #3b82f6;
        stroke-width: 2;
        fill: none;
    }
    
    .result-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35);
    }
    
    .result-label {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        color: #bfdbfe;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .result-label svg {
        width: 16px;
        height: 16px;
        stroke: #bfdbfe;
        stroke-width: 2.5;
        fill: none;
    }
    
    .result-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        text-transform: capitalize;
        margin-bottom: 0.3rem;
    }
    
    .result-confidence {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        color: #dbeafe;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .result-confidence svg {
        width: 16px;
        height: 16px;
        stroke: #dbeafe;
        stroke-width: 2.5;
        fill: none;
    }
    
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
    
    [data-testid="stImage"] img {
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Footer */
    .footer-text {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
    }
    
    .footer-text svg {
        width: 14px;
        height: 14px;
        stroke: #64748b;
        stroke-width: 2;
        fill: none;
    }
    
    .footer-text a {
        color: #3b82f6;
        text-decoration: none;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource(show_spinner=False)
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

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
st.markdown('''
<div class="header">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
    </svg>
    <h1 class="title">Image Classifier</h1>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Upload an image and let AI identify what\'s in it</div>', unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model, preprocess, tokenizer = load_model()
    text_features = encode_classes()

st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown('''
        <div class="section-label">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
            </svg>
            Uploaded Image
        </div>
        ''', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown('''
        <div class="section-label">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                <path d="M2 17l10 5 10-5"/>
                <path d="M2 12l10 5 10-5"/>
            </svg>
            Prediction
        </div>
        ''', unsafe_allow_html=True)
        
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
            <div class="result-label">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
                Detected
            </div>
            <div class="result-value">{pred_class}</div>
            <div class="result-confidence">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"/>
                    <line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
                Confidence: {confidence:.2f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('''
<div class="footer-text">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="16 18 22 12 16 6"/>
        <polyline points="8 6 2 12 8 18"/>
    </svg>
    Built by <a href="https://github.com/surajverma" target="_blank">Suraj Verma</a> • Powered by CLIP
</div>
''', unsafe_allow_html=True)
