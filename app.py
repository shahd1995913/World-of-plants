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
  import streamlit as st
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

st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .image-container {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .prediction-container {
        margin-bottom: 20px;
    }
    .class-container {
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .image-container {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .prediction-container {
        margin-bottom: 20px;
    }
    .class-container {
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import base64

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def main():
    st.title("Image Classification")
    st.markdown('<div class="title">Image Classification</div>', unsafe_allow_html=True)
    st.write("Upload an image for classification.")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True, clamp=True)
        st.markdown('<div class="image-container centered"><img src="data:image/png;base64,{}" width="300" /></div>'.format(image_to_base64(image)), unsafe_allow_html=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        st.markdown('<div class="prediction-container">Predicted Class: <strong>{}</strong></div>'.format(predicted_class_name), unsafe_allow_html=True)
        st.markdown('<div class="prediction-container">Confidence: <strong>{:.2f}%</strong></div>'.format(confidence), unsafe_allow_html=True)
        
        st.markdown('<div class="class-container">Other Classes:</div>', unsafe_allow_html=True)
        for i, class_name in enumerate(class_names):
            if i != predicted_class_index:
                st.markdown('<div class="class-container">{}: {:.2f}%</div>'.format(class_name, predictions[0][i] * 100), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
