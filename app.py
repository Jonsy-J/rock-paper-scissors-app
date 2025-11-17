import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random

# --- Load Keras model ---
MODEL_PATH = "rock_paper_scissors_3_class_mobilenet.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# --- Choices ---
choices = ['Rock', 'Paper', 'Scissors']
label_to_choice = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

# --- Helper functions ---
def preprocess_image(img: Image.Image):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def determine_winner(user_label, computer_label):
    result_code = (user_label - computer_label + 3) % 3
    if result_code == 0:
        return "It's a draw!"
    elif result_code == 1:
        return "You win! 🎉"
    else:
        return "Computer wins! 🤖"

# --- Streamlit UI ---
st.set_page_config(page_title="Rock-Paper-Scissors", page_icon="✊✋✌️", layout="centered")

# Custom background
st.markdown(
    """
    <style>
    .stApp {
        /* Dark background color */
        background-color: #1a1a1a;
        color: white;
    }

    /* Optional: make background image darker */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("https://images.unsplash.com/photo-1529070538774-1843cb3265df?fit=crop&w=1950&q=80") no-repeat center center;
        background-size: cover;
        filter: brightness(30%);
        z-index: -1;
    }

    /* Make text inside Streamlit readable */
    .stButton>button {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True

)

st.title("🎮 Rock-Paper-Scissors Game")
st.write("Upload an image of your hand (Rock, Paper, Scissors) or click a button below:")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload your hand image...", type=["jpg","jpeg","png"])
user_label = None
user_confidence = None

# --- Buttons ---
col1, col2, col3 = st.columns(3)
if col1.button("Rock"):
    user_label = 0
if col2.button("Paper"):
    user_label = 1
if col3.button("Scissors"):
    user_label = 2

# --- Handle uploaded image ---
if uploaded_file is not None:
    user_img = Image.open(uploaded_file).convert("RGB")
    st.image(user_img, caption="Your uploaded image", use_column_width=True)
    preprocessed = preprocess_image(user_img)
    predictions = model.predict(preprocessed, verbose=0)
    user_label = np.argmax(predictions)
    user_confidence = predictions[0][user_label]

# --- Play the game ---
if user_label is not None:
    user_choice_text = label_to_choice[user_label]
    if user_confidence is None:
        # Button press, assume 100% confidence
        user_confidence = 1.0

    st.write(f"Your choice: **{user_choice_text}** ({user_confidence*100:.2f}% confidence)")

    computer_label = random.randint(0, 2)
    computer_choice_text = label_to_choice[computer_label]
    st.write(f"Computer chose: **{computer_choice_text}**")

    result_text = determine_winner(user_label, computer_label)
    st.success(result_text)

