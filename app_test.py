
#Run "streamlit run app_test.py --server.enableXsrfProtection=false" in terminal to get website

import streamlit as st
import pickle
import numpy as np
import streamlit as st
import cv2

st.write("test")

def load_model():
    with open('95_model_1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

def show_predict_page():
    st.title("Pet Emotion Prediction")

    st.write("""###Upload picture of your pet here!""")
   
    image = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if img is not None:
        file_details = {"Filename":img.name,"FileType":img.type,"FileSize":img.size}
        st.write(file_details)
        image = Image.open(img)
        st.text("Original Image")
        st.image(image,use_column_width=True)
    ok = st.button("Submit")

show_predict_page()