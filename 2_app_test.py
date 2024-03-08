
import streamlit as st
import pickle
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


model = tf.keras.models.load_model('/mnt/c/Users/mattm/OneDrive/Documents/ecs111/ecs111_project_pet_emotions_repo/ecs111/2_95_model.h5')

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_emotion(img, model):
    # Resize and preprocess the image for your model
    #img = tf.keras.preprocessing.image.load_img(img, target_size=(300, 300))
    img = img.resize((300, 300)) 
    img_array =  tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img = img.resize((300, 300))  # Adjust target size to your model's expected input size
    #img_array = np.array(img) / 255.0  # Normalize the image array
    #img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the emotion
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    
    class_indices = {'Angry': 0, 'Other': 1, 'Sad': 2, 'happy': 3}
    class_labels = list(class_indices.keys())
    predicted_class_label = class_labels[predicted_class_index]
    #predictions = model.predict(img_array)
    #emotion_index = np.argmax(predictions)  # Assuming the model outputs class indices
    #emotions = ['Happy', 'Sad', 'Angry', 'Relaxed']  # Adjust according to your model
    return predicted_class_label
    #return emotions[emotion_index]

# Streamlit webpage layout
st.title('Pet Emotion Classifier')
st.write("Upload an image of your pet to classify its emotion.")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
class_labels = ['Angry', 'Other', 'Sad', 'Happy']


if uploaded_file is not None:
    image = load_image(uploaded_file)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption='Uploaded Image', use_column_width = True)

    with col3:
        st.write(' ')
    
    st.write("")
    st.write("Classifying...")
    label = predict_emotion(image, model)
    if label == 'Other':
        st.write("The pet is normal")
    else:
        st.write(f"The pet seems to be: {label}")
  