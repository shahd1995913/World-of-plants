import streamlit as st
st.title("Image Classification with Google's Teachable Machine")
st.header("Brain Tumor MRI Classification Example")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


class_names = [
    'Tomato blight disease',
    'Bacterial spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato mosaic virus',
    'Target Spot',
    'Powdery mildew',
    'Spider mites Two spotted spider mite'
]

model = keras.models.load_model('keras_model.h5')

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img = img.convert('RGB')  # Convert image to RGB format
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return keras.applications.mobilenet.preprocess_input(img_array)

def main():
   st.title("Image Classification")
   st.write("Upload an image for classification.")

   uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

   if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    st.write(f"Predicted Class: {predicted_class_name}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Compare with other classes
    st.write("Other Classes:")
    for i, class_name in enumerate(class_names):
        if i != predicted_class_index:
            st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")



if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
