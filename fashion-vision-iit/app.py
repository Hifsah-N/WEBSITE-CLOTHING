import streamlit as st
from PIL import Image
import os
import numpy as np
from utils import preprocess_image, extract_dominant_color, estimate_pattern, estimate_material, estimate_style
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import json
import hashlib

# --- Simple Authentication ---
USERS_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def register_form():
    st.title("📝 Sign Up for Fashion Vision IIT")
    with st.form("register_form"):
        username = st.text_input("Choose a username")
        password = st.text_input("Choose a password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register")
        if submitted:
            users = load_users()
            if not username or not password:
                st.error("Username and password required.")
            elif username in users:
                st.error("Username already exists.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                users[username] = hash_password(password)
                save_users(users)
                st.success("Registration successful! Please log in.")
                st.session_state.show_login = True
                st.rerun()

def login_form():
    st.title("🔒 Login to Fashion Vision IIT")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            users = load_users()
            if username in users and users[username] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

# --- MODERN BLACK & WHITE THEME CSS ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Poppins:wght@400;700&display=swap" rel="stylesheet">
<style>
body, .stApp {
    background: #fff !important;
    font-family: 'Poppins', 'Montserrat', sans-serif !important;
    color: #111 !important;
}

.bw-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 90vh;
}

.bw-card {
    background: rgba(240,240,240,0.92);
    border-radius: 28px;
    box-shadow: 0 0 32px #0002, 0 2px 8px #0001;
    border: 2px solid #222;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    max-width: 420px;
    width: 100%;
    margin: 2rem auto;
    position: relative;
    z-index: 2;
    color: #111;
    overflow: hidden;
}

.bw-logo {
    font-family: 'Montserrat', 'Poppins', sans-serif;
    font-size: 2.7rem;
    font-weight: 700;
    color: #111;
    letter-spacing: 2px;
    margin-bottom: 1.2rem;
    text-align: center;
    line-height: 1.1;
}

.bw-desc {
    color: #222;
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 2.2rem;
}

.stTextInput > div > input, .stTextInput input, .stTextArea textarea {
    background: #fff !important;
    color: #111 !important;
    border: 2px solid #222 !important;
    border-radius: 14px !important;
    font-size: 1.1rem !important;
    box-shadow: 0 0 8px #0001;
    padding: 0.8rem 1.2rem !important;
    margin-bottom: 1.1rem !important;
    transition: border 0.2s, box-shadow 0.2s;
}
.stTextInput > div > input:focus, .stTextInput input:focus, .stTextArea textarea:focus {
    border: 2px solid #111 !important;
    box-shadow: 0 0 16px #0002;
}

.stButton > button, .stForm button {
    font-family: 'Montserrat', 'Poppins', sans-serif !important;
    font-weight: 700;
    border-radius: 14px !important;
    background: #111 !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 2px 8px #0002;
    padding: 0.8rem 2.2rem !important;
    font-size: 1.1rem !important;
    transition: box-shadow 0.2s, background 0.2s, color 0.2s, transform 0.15s;
    cursor: pointer;
    outline: none !important;
    margin-top: 0.7rem;
}
.stButton > button:hover, .stForm button:hover {
    background: #fff !important;
    color: #111 !important;
    box-shadow: 0 0 16px #0003;
    border: 1.5px solid #111 !important;
    transform: scale(1.03);
}

.stForm label, .stTextInput label {
    color: #111 !important;
    font-weight: 600;
    letter-spacing: 0.5px;
    font-size: 1.05rem;
    margin-bottom: 0.3rem;
}

.stForm .stAlert {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# --- AUTH PAGE: Neon Centered Card ---
def neon_auth_container(content_func):
    st.markdown('<div class="bw-center"><div class="bw-card">', unsafe_allow_html=True)
    st.markdown('<div class="bw-logo">FASHION VISION<br>IIT</div>', unsafe_allow_html=True)
    st.markdown('<div class="bw-desc">AI-powered fashion detection and analysis.<br>Sign in or create an account to get started!</div>', unsafe_allow_html=True)
    content_func()
    st.markdown('</div></div>', unsafe_allow_html=True)

# --- Auth Page Toggle ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = True

if not st.session_state.logged_in:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", type="primary"):
            st.session_state.show_login = True
    with col2:
        if st.button("Sign Up"):
            st.session_state.show_login = False
    if st.session_state.show_login:
        neon_auth_container(login_form)
    else:
        neon_auth_container(register_form)
    st.stop()

# --- Main App (only visible if logged in) ---
st.set_page_config(page_title="Fashion Vision IIT", layout="wide")

st.title("👗 Fashion Vision IIT")
st.markdown("AI-powered fashion detection and analysis app.")

# --- Logout Button ---
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

@st.cache_resource
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload an image of clothing (JPG/PNG)", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    st.markdown("#### Try with a sample image:")
    sample_path = os.path.join("assets", "example.jpg")
    if os.path.exists(sample_path):
        image = Image.open(sample_path)
        st.image(image, caption="Sample Image", use_column_width=True)

if image:
    st.markdown("### 📝 Detected Fashion Attributes")
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    clothing_type, description, confidence = decoded[0][1], decoded[0][1], float(decoded[0][2])
    color = extract_dominant_color(image)
    pattern = estimate_pattern(image)
    material = estimate_material(image)
    style = estimate_style(image)
    tags = ["elegant", "office", "minimalist"]
    # Styled card
    st.markdown(f"""
    <div class="result-card">
        <h4 style='margin-bottom:0.5rem;'>👕 Clothing Type: <b>{clothing_type.title()}</b></h4>
        <div><b>Pattern:</b> {pattern}</div>
        <div><b>Material:</b> {material}</div>
        <div><b>Style:</b> {style}</div>
        <div><b>Color:</b> <span class='color-swatch' style='background:{color['hex']}'></span> {color['name']} ({color['hex']}, {color['rgb']})</div>
        <div><b>Confidence:</b> {confidence:.2%}
            <div class='confidence-bar' style='width:{int(confidence*100)}%;background:linear-gradient(90deg,#6366f1 0%,#a5b4fc 100%);'></div>
        </div>
        <div><b>Tags:</b> {' '.join([f'<span style=\'background:#e0e7ff;padding:2px 8px;border-radius:8px;margin-right:4px;\'>{t}</span>' for t in tags])}</div>
    </div>
    """, unsafe_allow_html=True)
    # JSON output
    st.markdown("#### JSON Output")
    result_json = {
        "item": clothing_type.title(),
        "color": color,
        "pattern": pattern,
        "material": material,
        "style": style,
        "confidence": confidence,
        "tags": tags
    }
    st.json(result_json)
    st.download_button("Download JSON", data=json.dumps(result_json, indent=2), file_name="fashion_result.json", mime="application/json")
else:
    st.info("Upload an image to analyze fashion attributes.")

# Feedback section
st.markdown("---")
st.markdown("### ⭐ Feedback")

with st.form("feedback_form"):
    stars = st.slider("How would you rate this app?", 1, 5, 5)
    comment = st.text_area("Any comments or suggestions?", "")
    submitted = st.form_submit_button("Submit Feedback")
    if submitted:
        feedback_entry = {"stars": stars, "comment": comment}
        feedback_path = os.path.join("feedback.json")
        # Load existing feedback
        if os.path.exists(feedback_path):
            with open(feedback_path, "r", encoding="utf-8") as f:
                try:
                    feedback_data = json.load(f)
                except Exception:
                    feedback_data = []
        else:
            feedback_data = []
        feedback_data.append(feedback_entry)
        with open(feedback_path, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)
        st.success("Thank you for your feedback!")
