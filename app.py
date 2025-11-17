import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os

# --- Load Keras model ---
MODEL_PATH = 'rock_paper_scissors_3_class_mobilenet.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# --- Choices and mappings ---
choices = ['Rock', 'Paper', 'Scissors']
label_to_choice = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

# --- Preprocess function ---
def preprocess_image(img: Image.Image):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# --- App UI ---
st.title("🎮 Rock-Paper-Scissors Game")
st.write("Upload an image of your hand: Rock, Paper, or Scissors.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    user_img = Image.open(uploaded_file).convert("RGB")
    st.image(user_img, caption="Your uploaded image", use_column_width=True)

    # Preprocess and predict
    preprocessed = preprocess_image(user_img)
    user_pred_label = np.argmax(model.predict(preprocessed, verbose=0))
    user_pred_text = label_to_choice[user_pred_label]
    st.write(f"Model prediction for your image: **{user_pred_text}**")

    # Computer choice
    computer_choice_text = random.choice(choices)
    st.write(f"Computer chose: **{computer_choice_text}**")
    computer_label = choices.index(computer_choice_text)

    # Determine winner
    result_code = (user_pred_label - computer_label + 3) % 3
    if result_code == 0:
        result_text = "It's a draw!"
    elif result_code == 1:
        result_text = "You win! 🎉"
    else:
        result_text = "Computer wins! 🤖"

    st.write(f"🎯 {result_text}")
