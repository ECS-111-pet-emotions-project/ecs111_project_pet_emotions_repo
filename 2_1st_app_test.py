
import streamlit as st
import pickle
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


model = tf.keras.models.load_model('/Users/alyssahharnden/Downloads/ECS_111_Project/95_model.h5')

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

#Streamlit webpage layout
#Apply custom CSS for background color
st.markdown(
    """
    <style>
        .stApp {
            background-color: #add8e6; /* Set the background color to light pastel blue */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# uwu cat image printed on webpage
uwu_cat_image_path = "/Users/alyssahharnden/Downloads/ECS_111_Project/uwu_cat.jpeg"
uwu_cat_image = Image.open(uwu_cat_image_path)

# Adjust width and position of the image
uwu_image_column, uwu_content_column = st.columns([1, 2.3])  # Create two columns, adjust width as needed
with uwu_image_column:
    st.image(uwu_cat_image, use_column_width=True)  # Adjust width as needed

# Title & description
with uwu_content_column:
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=DynaPuff&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown('<h1 style="font-family: \'DynaPuff\'; font-size: 52px;">Pet Feelz 🐾</h1>', unsafe_allow_html=True)
    st.write("Welcome to Pet Feelz!")
    st.write("Here, you can upload an image of your adorable pet and receive suggestions for improving or maintaining your **pet's 'feelz'** based on the emotion that we've detected!")


# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image!", type=["png", "jpg", "jpeg"])
class_labels = ['Angry', 'Other', 'Sad', 'happy']


if uploaded_file is not None:
    image = load_image(uploaded_file)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption='Uploaded Image', use_column_width = True)

    with col3:
        st.write(' ')

    #Creating placeholder such that we can update "classifying -> classified"
    st.write("")
    classification_placeholder = st.empty()
    classification_placeholder.write("Classifying...")

    #Performing classification
    label = predict_emotion(image, model)

    #Updating placeholder post-classification
    classification_placeholder.write("Classified!")

    #Results + suggestions
    if label == 'Other':
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>CONTENT! 😌</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest continuing doing what you're doing so your pet remains comfortable!</b>
        </div>
        """, unsafe_allow_html=True)
    elif label == 'Angry':
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>ANGRY! 😠</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest giving your pet some space to calm down and maybe a treat too!</b>
        </div>
        """, unsafe_allow_html=True)
    elif label == 'Sad':
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>SAD! 🥺</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest providing your pet with comfort and lots of love!</b>
        </div>
        """, unsafe_allow_html=True)
    elif label == 'happy':
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>HAPPY! 🤗</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest giving your pet some extra cuddles and playtime!</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        print("")
        
# Copyright
st.write(" ")
st.write("© 2024 MRL. All rights reserved.")
  
