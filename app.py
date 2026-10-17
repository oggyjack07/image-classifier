import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# ---------- Page Config ----------
st.set_page_config(
    page_title="Image Classifier | Suraj Verma",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%233b82f6'><rect x='3' y='3' width='18' height='18' rx='2'/><circle cx='8.5' cy='8.5' r='1.5' fill='white'/><polyline points='21 15 16 10 5 21' stroke='white' stroke-width='2' fill='none'/></svg>",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- CSS ----------
st.markdown("""
<style>
    .stApp {
        background: #0a0f1c;
    }
    
    header, footer, #MainMenu {visibility: hidden;}
    
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 640px;
    }
    
    /* Header */
    .header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
    .header svg {
        width: 34px;
        height: 34px;
        stroke: #3b82f6;
        stroke-width: 2;
        fill: none;
    }
    .title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .subtitle {
        font-size: 0.9rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Main card */
    .main-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        margin-bottom: 1.5rem;
    }
    
    /* Compact uploader */
    [data-testid="stFileUploader"] {
        background: #0a0f1c;
        border: 1.5px dashed #334155;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        transition: all 0.25s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #0f172a;
    }
    [data-testid="stFileUploader"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stFileUploader"] > div {
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0.5rem !important;
        min-height: auto !important;
    }
    
    /* Result panel - right side */
    .result-panel {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 200px;
    }
    .result-label {
        color: #bfdbfe;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .result-value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 800;
        text-transform: capitalize;
        line-height: 1.2;
        word-break: break-word;
    }
    
    /* Image wrapper - left side */
    .image-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        min-height: 200px;
    }
    .image-wrapper img {
        max-width: 100%;
        max-height: 200px;
        width: auto;
        height: auto;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border: 1px solid #1f2937;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 1.8rem;
        padding-top: 1.2rem;
        border-top: 1px solid #1f2937;
    }
    .footer-text {
        color: #475569;
        font-size: 0.75rem;
        margin-bottom: 0.9rem;
    }
    .social-links {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        flex-wrap: wrap;
    }
    .social-link {
        display: flex;
        align-items: center;
        gap: 5px;
        color: #94a3b8;
        text-decoration: none;
        font-size: 0.8rem;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    .social-link:hover {
        color: #3b82f6;
    }
    .social-link svg {
        width: 14px;
        height: 14px;
        stroke: currentColor;
        stroke-width: 2;
        fill: none;
    }
    
    /* Mobile */
    @media (max-width: 640px) {
        .main .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .title { font-size: 1.4rem; }
        .header svg { width: 28px; height: 28px; }
        .result-value { font-size: 1.3rem; }
    }
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
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

# ---------- Header ----------
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

# ---------- Load Model ----------
with st.spinner("Loading model..."):
    model, preprocess, tokenizer = load_model()
    text_features = encode_classes()

# ---------- Uploader Card ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Result Area: Left Image, Right Result ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("Classifying..."):
        img = image.resize((224, 224))
        img_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        pred_class = classes[similarity.argmax().item()]
    
    # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown(f'''
        <div class="image-wrapper">
            <img src="data:image/png;base64,{img_str}" />
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="result-panel">
            <div class="result-label">Detected</div>
            <div class="result-value">{pred_class}</div>
        </div>
        ''', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown('''
<div class="footer">
    <div class="footer-text">Built by Suraj Verma</div>
    <div class="social-links">
        <a class="social-link" href="https://github.com/oggyjack07" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
            </svg>
            GitHub
        </a>
        <a class="social-link" href="https://ayuushss-portfolio.vercel.app/" target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="2" y1="12" x2="22" y2="12"/>
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
            </svg>
            Portfolio
        </a>
        <a class="social-link" href="mailto:kisycoless@gmail.com">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                <polyline points="22,6 12,13 2,6"/>
            </svg>
            Email
        </a>
    </div>
</div>
''', unsafe_allow_html=True)
