import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os

# --- Page configuration ---
st.set_page_config(
    page_title="Rock-Paper-Scissors Classifier",
    page_icon="✊✋✌️",
    layout="centered",
)

# --- Header ---
st.title("✊✋✌️ Rock-Paper-Scissors Classifier ✌️✋✊")
st.write("Upload an image of your hand to play against the computer.")

# --- Load TFLite model ---
MODEL_PATH = 'rock_paper_scissors_quantized.tflite'

if os.path.exists(MODEL_PATH):
    st.success("Model loaded successfully!")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
else:
    st.error(f"Model file '{MODEL_PATH}' not found. Please upload it.")
    st.stop()  # Stop execution if model is missing

# --- Choices ---
choices = ['Rock', 'Paper', 'Scissors']
choice_to_label = {'Rock': 0, 'Paper': 1, 'Scissors': 2}
label_to_choice = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

# --- Helper functions ---
def preprocess_image(image):
    img = image.resize((150, 150))
    image_array = np.array(img)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(input_dtype)
    return image_array

def make_prediction(preprocessed_input):
    interpreter.set_tensor(input_details[0]['index'], preprocessed_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_label = np.argmax(output)
    confidence_scores = output
    return predicted_label, confidence_scores

def determine_winner(user_label, computer_label):
    result_code = (user_label - computer_label + 3) % 3
    if result_code == 0:
        return "It's a draw!"
    elif result_code == 1:
        return "You win!"
    else:
        return "Computer wins!"

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image of your hand (jpg, png)...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    user_image = Image.open(uploaded_file).convert("RGB")
    st.image(user_image, caption="Your uploaded image", use_column_width=True)

    if st.button("Play"):
        # Preprocess user image and predict
        preprocessed_user = preprocess_image(user_image)
        user_pred_label, user_confidences = make_prediction(preprocessed_user)
        user_pred_text = label_to_choice[user_pred_label]

        # Random computer choice
        computer_pred_text = random.choice(choices)
        computer_label = choice_to_label[computer_pred_text]

        # Display predictions
        st.subheader("Predictions")
        st.write(f"**Your hand prediction:** {user_pred_text}")
        st.write("Confidence scores:")
        for i, choice in enumerate(choices):
            st.write(f"{choice}: {user_confidences[i]:.2f}")

        st.write(f"**Computer choice:** {computer_pred_text}")

        # Determine winner
        st.success(determine_winner(user_pred_label, computer_label))
