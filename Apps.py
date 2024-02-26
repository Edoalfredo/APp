
import streamlit as st
import numpy as np
import tensorflow as tf
from IPython.display import display, Image
from keras.preprocessing import image, image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.backend import clear_session


picture = st.camera_input("Take a picture")

if picture:
    # Convert UploadedFile to string path
    model = load_model("mdl85.h5")
    file_path = picture

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class_index = np.argmax(predictions[0])
    print('class index is:', predicted_class_index)

    if(predicted_class_index==0):
        st.write("Batik Bali")
    if(predicted_class_index==1):
        st.write("Batik Betawi") 
    if(predicted_class_index==2):
        st.write("Batik Cendrawasih")
    if(predicted_class_index==3):
        st.write("Batik Dayak")
    if(predicted_class_index==4):
        st.write("Ikat Geblek Renteng")
    if(predicted_class_index==5):
        st.write("Batik Ikat Celup")
    if(predicted_class_index==6):
        st.write("Batik Insang")
    if(predicted_class_index==7):
        st.write("Batik Kawung")
    if(predicted_class_index==8):
        st.write("Batik Lasem")      
    if(predicted_class_index==9):
        st.write("Batik Megamendung")
    if(predicted_class_index==10):
        st.write("Batik Pala")
    # Display the uploaded image and prediction
    st.image(picture)