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
    # Set page width and center content
    max_width = 800
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin: 0 auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Image Classification")
    st.subheader("Upload an image for classification")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class_index = predictions.argmax()
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        # Display predicted class and confidence
        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Predicted Class</h2>
                <h3 style='font-weight: bold; color: #0072B2;'>{}</h3>
            </div>
            """.format(predicted_class_name),
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Confidence</h2>
                <h3 style='font-weight: bold; color: #0072B2;'>{:.2f}%</h3>
            </div>
            """.format(confidence),
            unsafe_allow_html=True
        )

        # Display other classes
        st.markdown(
            """
            <div style='text-align: center;'>
                <h2 style='font-weight: bold; color: #0072B2;'>Other Classes</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        for i, class_name in enumerate(class_names):
            if i != predicted_class_index:
                st.markdown(
                    """
                    <div style='text-align: center;'>
                        <h3 style='font-weight: bold; color: #0072B2;'>{}: {:.2f}%</h3>
                    </div>
                    """.format(class_name, predictions[0][i] * 100),
                    unsafe_allow_html=True
                )


if __name__ == '__main__':
    main()
