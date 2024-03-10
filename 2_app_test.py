
import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input




model = tf.keras.models.load_model('/mnt/c/Users/mattm/OneDrive/Documents/ecs111/ecs111_project_pet_emotions_repo/ecs111/95_model.h5')
ds_name = 'Pets Facial Expression'
data_dir = 'pet_images'

def gen_data_and_labels(data_dir):
    #Identifies and stores filepaths to images in images variables
    #Labels images according to the file they were found in.
    
    image_paths = []
    labels = []

    files = os.listdir(data_dir)
    for file in files:
        
        if file == 'Master Folder':
            continue
            
        filepath = os.path.join(data_dir, file)
        
        imagelist = os.listdir(filepath)
      
        for im in imagelist:
           
            im_path = os.path.join(filepath, im)
            image_paths.append(im_path)
            labels.append(file)
            
    return image_paths, labels


image_paths, labels = gen_data_and_labels(data_dir)

def create_df(image_paths, labels):

    Fseries = pd.Series(image_paths, name= 'image_paths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis= 1)
    return df

df = create_df(image_paths, labels)

Happy = df.groupby("labels").get_group("happy")
Sad = df.groupby("labels").get_group("Sad")
Angry = df.groupby("labels").get_group("Angry")
Other = df.groupby("labels").get_group("Other")

def plot_emotions(emotion):
    fig, axes = plt.subplots(ncols = 5,nrows = 1, figsize=(20, 20))
    for i in range(0,5):
        index = random.sample(range(len(emotion)),1)
        index = int(''.join(map(str, index)))

        filename = emotion.iloc[index]["image_paths"]
        label = emotion.iloc[index]["labels"]
        image = Image.open(filename)
        axes[i].imshow(image)
        axes[i].set_title("Labels: " + label, fontsize = 30)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
    

    plt.show()
    st.pyplot(fig)
    
 
grouped_data = df.groupby("labels")
num_images_per_category = 5

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
    acc = np.max(prediction)
    class_indices = {'Angry': 0, 'Other': 1, 'Sad': 2, 'happy': 3}
    class_labels = list(class_indices.keys())
    predicted_class_label = class_labels[predicted_class_index]
    #predictions = model.predict(img_array)
    #emotion_index = np.argmax(predictions)  # Assuming the model outputs class indices
    #emotions = ['Happy', 'Sad', 'Angry', 'Relaxed']  # Adjust according to your model
    return predicted_class_label, acc
    #return emotions[emotion_index]

# Streamlit webpage layout
st.title('Pet Emotion Classifier')
st.write("Upload an image of your pet to classify its emotion.")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
class_labels = ['Angry', 'Other', 'Sad', 'Happy']

def show_similar_img(label):
    if label == "Angry":
        plot_emotions(Angry)
    elif label == "Other":
        plot_emotions(Other)
    elif label == "Sad":
        plot_emotions(Sad)
    elif label == "Happy":
        plot_emotions(Happy)

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
    label, acc = predict_emotion(image, model)
    acc = "{:.2f}".format(100*acc)
    if label == 'Other':
        st.write(f"With a probability of {acc}%, your pet appears to be neither angry, happy, or sad.")
    else:
        st.write(f"With a probability of {acc}%, your pet appears to be {label}.")
    
    st.write(f"Other {label} pets look like this" )
    show_similar_img(label)