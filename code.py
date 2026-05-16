import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model you saved from the notebook
model = tf.keras.models.load_model('flower_model.keras')
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

st.title("Flower Classification App")
st.write("Upload a flower image and the model will predict its type.")

# 2. Upload Image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 3. Preprocess the image to match notebook settings (180x180)
    img = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # 4. Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.success(f"This image most likely belongs to {class_names[np.argmax(score)]} "
               f"with a {100 * np.max(score):.2f}% confidence.")