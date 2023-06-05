import streamlit as st
st.title("Image Classification")
st.header("Upload a Image")
st.text("Done by shahed Alhateeb 2023")
import cv2
from keras.models import load_model
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

    # Set the overall page width
    st.markdown(
        """
        <style>
        .reportview-container .main {
            max-width: 800px;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create columns for the image and predictions
    col1, col2 = st.columns([1, 1.5])

    col1.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='color: #f63366;'>Image Classification</h1>
            <p>Upload an image for classification.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = col1.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        col2.write("Predicted Class:")
        col2.write(predicted_class_name)

        col2.write("Confidence:")
        col2.write("{:.2f}%".format(confidence))

        col2.write("Other Classes:")
        for i, class_name in enumerate(class_names):
            if i != predicted_class_index:
                col2.write("{}: {:.2f}%".format(class_name, predictions[0][i] * 100))



if __name__ == '__main__':
    main()
