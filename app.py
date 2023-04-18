import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tempfile import NamedTemporaryFile


st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def loading_model():
    model = "saved_model/full_tuned_model-01.h5"
    model_loader = load_model(model)
    return model_loader


model = loading_model()

st.title("Pneumonia Classification using Computer Vision and Transfer Learning.")

st.subheader("This application leverages VGG-16 pre-trained model to predict " +
             "whether an X-ray image is positive or negative for Pneumonia.")

background = Image.open('background-image/background-image.jpg')
st.image(background, use_column_width=True)
st.write(
    "Image uploaded from [unplash](https://unsplash.com/photos/2-Hv-Rg4asA)")

image_file = st.file_uploader(
    "Upload an image file: ", type=["jpg", "jpeg", "png"])

buffer = image_file
temp_file = NamedTemporaryFile(delete=False)

# make prediction with an upload image


def make_prediction():
    if buffer is not None:
        temp_file.write(buffer.getvalue())
        st.write(image.load_img(temp_file.name))

    if buffer is None:
        st.write("After the X-ray image is done uploading and reshaping.  Please hit the Predict " +
                 "button to check whether the image belongs to a pneumonia person or not.")

    else:
        new_img = image.load_img(
            temp_file.name, target_size=(224, 224), color_mode='rgb')

        # Preprocessing the image
        preprocess_img = image.img_to_array(new_img)
        preprocess_img = preprocess_img/255
        preprocess_img = np.expand_dims(preprocess_img, axis=0)
        # display image
        st.image(preprocess_img, channels="RGB")
        st.write('Image type: ', type(new_img))
        st.write('Image size: ', new_img.size)

    # create predicition button
    generate_pred = st.button("Generate Prediction")

    if buffer is None:
        if generate_pred:
            st.write("Please upload an X-ray image. or Try again.")

    else:
        if buffer is not None:
            if generate_pred:
                THRESHOLD = 0.5
                output = model.predict(preprocess_img)[0][0]
                st.write('Prediction Output:', output)

                prediction = 1 if (output > THRESHOLD) else 0

                if prediction > THRESHOLD:
                    st.write(
                        "Prediction: The X-ray image belongs to a PNEUMONIA person.\n")
                else:
                    st.write(
                        "Prediction: The X-ray image belongs to a NORMAL person.\n")
                    
              CLASSES = ['NORMAL', 'PNEUMONIA']

              ClassPred = CLASSES[prediction]
              ClassProb = output

              print("Predicition", ClassPred)
              print("Prob: {:.2%}".format(ClassProb))


        

    
    
    
if __name__ == '__main__':
    make_prediction()
