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

# --- Load TFLite model ---
MODEL_PATH = 'rock_paper_scissors_quantized.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# --- Choices ---
choices = ['Rock', 'Paper', 'Scissors']
choice_to_label = {'Rock': 0, 'Paper': 1, 'Scissors': 2}
label_to_choice = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

# --- Helper functions ---
def preprocess_image(image):
    """Resize, normalize, and expand dims for TFLite model."""
    img = image.resize((150, 150))
    image_array = np.array(img)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(input_dtype)
    return image_array

def make_prediction(preprocessed_input):
    """Run TFLite model and return predicted label and confidence scores."""
    interpreter.set_tensor(input_details[0]['index'], preprocessed_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_label = np.argmax(output)
    confidence_scores = output
    return predicted_label, confidence_scores

def determine_winner(user_label, computer_label):
    """Determine winner based on standard Rock-Paper-Scissors rules."""
    result_code = (user_label - computer_label + 3) % 3
    if result_code == 0:
        return
