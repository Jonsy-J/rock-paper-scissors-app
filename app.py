import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os

# --- Page config ---
st.set_page_config(
    page_title="Rock-Paper-Scissors Game",
    page_icon="✊✋✌️",
    layout="centered"
)

# --- Load Keras model ---
MODEL_PATH = 'rock_paper_scissors_3_class_mobilenet.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# --- Choices mapping ---
choices = ['Rock', 'Paper', 'Scissors']
choice_to_label = {'Rock': 0, 'Paper': 1, 'Scissors': 2}
label_to_choice = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

# --- Helper functions ---
def preprocess_image(image_array):
    if not isinstance(image_array, np.ndarray):
        image_array = np.array(image_array)
    if image_array.shape[0] != 150 or image_array.shape[1] != 150:
        img = Image.fromarray(image_array.astype(np.uint8))
        img = img.resize((150, 150))
        image_array = np.array(img)
    image = image_array / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

def make_prediction(preprocessed_input):
    output = model.predict(preprocessed_input)
    return np.argmax(output)

def create_dummy_image(label):
    dummy_img = np.zeros((150, 150, 3), dtype=np.uint8)
    if label == 0:
        dummy_img[:,:] = [0,0,0]        # Rock
    elif label == 1:
        dummy_img[:,:] = [255,255,255]  # Paper
    elif label == 2:
        dummy_img[:,:] = [127,127,127]  # Scissors
    return dummy_img

# --- App UI ---
st.title("🎮 Rock-Paper-Scissors Game")
st.sidebar.title("Model Info")
file_size_bytes = os.path.getsize(MODEL_PATH)
file_size_mb = file_size_bytes / (1024*1024)
st.sidebar.write(f"Keras model: {MODEL_PATH}")
st.sidebar.write(f"Size: {file_size_mb:.2f} MB")

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("✊ Rock"):
        st.session_state.choice = "Rock"
with col2:
    if st.button("✋ Paper"):
        st.session_state.choice = "Paper"
with col3:
    if st.button("✌️ Scissors"):
        st.session_state.choice = "Scissors"

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Game logic
if 'choice' in st.session_state:
    user_choice_text = st.session_state.choice
    user_label = choice_to_label[user_choice_text]

    computer_choice_text = random.choice(choices)
    computer_label = choice_to_label[computer_choice_text]

    # Dummy images and model predictions
    user_pred_label = make_prediction(preprocess_image(create_dummy_image(user_label)))
    comp_pred_label = make_prediction(preprocess_image(create_dummy_image(computer_label)))
    user_pred_text = label_to_choice[user_pred_label]
    comp_pred_text = label_to_choice[comp_pred_label]

    # Determine winner
    result_code = (user_label - computer_label + 3) % 3
    if result_code == 0:
        result_text = "It's a draw!"
    elif result_code == 1:
        result_text = "You win! 🎉"
    else:
        result_text = "Computer wins! 🤖"

    # Store history
    st.session_state.history.append({
        'You': user_choice_text,
        'Computer': computer_choice_text,
        'Result': result_text
    })

    # Display round results
    st.subheader("Round Results")
    st.write(f"Your choice: **{user_choice_text}**")
    st.write(f"Computer choice: **{computer_choice_text}**")
    st.write(f"Model prediction for your dummy input: **{user_pred_text}**")
    st.write(f"Model prediction for computer dummy input: **{comp_pred_text}**")
    st.write(f"🎯 {result_text}")

    # Display history
    st.subheader("Game History")
    st.table(st.session_state.history)
