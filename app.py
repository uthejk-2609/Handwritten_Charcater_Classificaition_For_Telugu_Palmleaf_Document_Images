import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64

# Define a function to add a background image
def add_bg_image(image_path):

    with open(image_path, "rb") as file:
        img_data = file.read()
    
    encoded_img = base64.b64encode(img_data).decode()

    style = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        width: 100vw;
    }
    </style>
    ''' % encoded_img

    st.markdown(style, unsafe_allow_html=True)

data_dir = r"C:\Amrita\Deep Learning\DATA_LABELS_FINAL"
categories = os.listdir(data_dir)

with open(r"C:\Amrita\Deep Learning\Codes\CNN\CNN_model.json", "r") as json_file:
    loaded_model_json_cnn = json_file.read()

loaded_model_cnn = model_from_json(loaded_model_json_cnn)
loaded_model_cnn.load_weights(r"C:\Amrita\Deep Learning\Codes\CNN\CNN_model.h5")

with open(r"C:\Amrita\Deep Learning\Codes\LSTM\LSTM_model.json", "r") as json_file:
    loaded_model_json_lstm = json_file.read()

loaded_model_lstm = model_from_json(loaded_model_json_lstm)
loaded_model_lstm.load_weights(r"C:\Amrita\Deep Learning\Codes\LSTM\LSTM_model.h5")

with open(r"C:\Amrita\Deep Learning\Codes\Simple RNN\Simple_RNN_model.json", "r") as json_file:
    loaded_model_json_rnn = json_file.read()

loaded_model_rnn = model_from_json(loaded_model_json_rnn)
loaded_model_rnn.load_weights(r"C:\Amrita\Deep Learning\Codes\Simple RNN\Simple_RNN_model.h5")

with open(r"C:\Amrita\Deep Learning\Codes\LeNet-5\LeNet5_model.json", "r") as json_file:
    loaded_model_json_lenet5 = json_file.read()

loaded_model_lenet5 = model_from_json(loaded_model_json_lenet5)
loaded_model_lenet5.load_weights(r"C:\Amrita\Deep Learning\Codes\LeNet-5\LeNet5_model.h5")

with open(r"C:\Amrita\Deep Learning\Codes\Alex Net\AlexNet.json", "r") as json_file:
    loaded_model_json_alexnet = json_file.read()

loaded_model_alexnet = model_from_json(loaded_model_json_alexnet)
loaded_model_alexnet.load_weights(r"C:\Amrita\Deep Learning\Codes\Alex Net\AlexNet.h5")

def predict_image_cnn(image_path):
    img = image.load_img(image_path, target_size=(100, 100), color_mode="grayscale")
    img = img.convert("L")

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = loaded_model_cnn.predict(img_array)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class_index]

    st.subheader("Predicted Category: " + predicted_category)

def predict_image_lstm(image_path):
    img = image.load_img(image_path, target_size=(32, 32), color_mode="grayscale")
    img = img.convert("L")

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = loaded_model_lstm.predict(img_array)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class_index]

    st.subheader("Predicted Category: " + predicted_category)

def predict_image_rnn(image_path):
    img = image.load_img(image_path, target_size=(32, 32), color_mode="grayscale")
    img = img.convert("L")

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = loaded_model_rnn.predict(img_array)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class_index]

    st.subheader("Predicted Category: " + predicted_category)

def predict_image_lenet5(image_path):
    img = image.load_img(image_path, target_size=(32, 32), color_mode="grayscale")
    img = img.convert("L")

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = loaded_model_lenet5.predict(img_array)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class_index]

    st.subheader("Predicted Category: " + predicted_category)

def predict_image_alexnet(image_path):
    img = image.load_img(image_path, target_size=(227, 227), color_mode="grayscale")
    img = img.convert("L")
    
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
        
    predictions = loaded_model_alexnet.predict(img_array)
        
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class_index]
    
    st.subheader("Predicted Category: " + predicted_category)

def main():
    CNN_confusion_matrix = Image.open(r'C:\Amrita\Deep Learning\Codes\images\CNN_Confusion_Matrix.png')
    LSTM_confusion_matrix = Image.open(r'C:\Amrita\Deep Learning\Codes\images\LSTM_Confusion_Matrix.png')
    Simple_RNN_confusion_matrix = Image.open(r'C:\Amrita\Deep Learning\Codes\images\Simple_RNN_Confusion_Matrix.png')
    LeNet5_confusion_matrix = Image.open(r'C:\Amrita\Deep Learning\Codes\images\LeNet5_Confusion_Matrix.png')
    AlexNet_confusion_matrix = Image.open(r'C:\Amrita\Deep Learning\Codes\images\AlexNet_Confusion_Matrix.png')

    st.sidebar.title("Select Model")
    st.title("Character Classification")
    

    model_selection = st.sidebar.selectbox("Select Model", ("CNN", "LSTM", "Simple RNN", "LeNet-5", "AlexNet"))
    

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = "./temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Predict"):
            if model_selection == "CNN":
                predict_image_cnn(image_path)
                st.image(CNN_confusion_matrix, caption='Confusion Matrix of Custom CNN', width=300, use_column_width=True)
            elif model_selection == "LSTM":
                predict_image_lstm(image_path)
                st.image(LSTM_confusion_matrix, caption='Confusion Matrix of LSTM', width=300, use_column_width=True)
            elif model_selection == "Simple RNN":
                predict_image_rnn(image_path)
                st.image(Simple_RNN_confusion_matrix, caption='Confusion Matrix of Simple RNN', width=300, use_column_width=True)
            elif model_selection == "LeNet-5":
                predict_image_lenet5(image_path)
                st.image(LeNet5_confusion_matrix, caption='Confusion Matrix of LeNet-5', width=300, use_column_width=True)
            elif model_selection == "AlexNet":
                predict_image_alexnet(image_path)
                st.image(AlexNet_confusion_matrix, caption='Confusion Matrix of AlexNet', width=300, use_column_width=True)

add_bg_image(r"C:\Amrita\Deep Learning\Codes\images\bg1.jpg")




# Run the app
if __name__ == "__main__":
    main()

















