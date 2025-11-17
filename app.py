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
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Background styling ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #74ebd5, #ACB6E5);
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 120px;
        font-size: 18px;
        border-radius: 10px;
        margin: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stAlert {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Load TFLite model ---
MODEL_PATH = 'rock_paper_scissors_quantized_mobilenet.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']

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
    return image.astype(input_dtype)

def make_prediction(preprocessed_input):
    interpreter.set_tensor(input_details[0]['index'], preprocessed_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

def create_dummy_image(label):
    dummy_img = np.zeros((150, 150, 3), dtype=np.uint8)
    if label == 0:
        dummy_img[:,:] = [0,0,0]        # Rock - black
    elif label == 1:
        dummy_img[:,:] = [255,255,255]  # Paper - white
    elif label == 2:
        dummy_img[:,:] = [127,127,127]  # Scissors - grey
    return dummy_img

# --- Sidebar Info ---
st.sidebar.title("Model Info")
file_size_bytes = os.path.getsize(MODEL_PATH)
file_size_mb = file_size_bytes / (1024 * 1024)
st.sidebar.write(f"TFLite model: **{MODEL_PATH}**")
st.sidebar.write(f"Size: **{file_size_mb:.2f} MB**")
st.sidebar.write("Uses MobileNetV2 feature extractor and dynamic range quantization.")
st.sidebar.write("---")

# --- App Title ---
st.title("🎮 Rock-Paper-Scissors Game")
st.markdown("Choose your move below and see if you can beat the computer!")

# --- Initialize session state ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Buttons for player choice ---
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

# --- Game logic ---
if 'choice' in st.session_state:
    user_choice_text = st.session_state.choice
    user_label = choice_to_label[user_choice_text]

    computer_choice_text = random.choice(choices)
    computer_label = choice_to_label[computer_choice_text]

    # Create dummy images and get predictions
    user_dummy_img = create_dummy_image(user_label)
    user_pred_label = make_prediction(preprocess_image(user_dummy_img))
    user_pred_text = label_to_choice[user_pred_label]

    comp_dummy_img = create_dummy_image(computer_label)
    comp_pred_label = make_prediction(preprocess_image(comp_dummy_img))
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

    # --- Show round info ---
    st.subheader("Round Results")
    st.write(f"Your choice: **{user_choice_text}**")
    st.write(f"Computer choice: **{computer_choice_text}**")
    st.write(f"Model prediction for your dummy input: **{user_pred_text}**")
    st.write(f"Model prediction for computer dummy input: **{comp_pred_text}**")
    st.write(f"🎯 {result_text}")

    # --- Show history ---
    st.subheader("Game History")
    st.table(st.session_state.history)
