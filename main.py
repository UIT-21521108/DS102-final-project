import streamlit as st
import pickle as pkl
import numpy as np
import sklearn
from PIL import Image
import requests
from io import BytesIO

class_list = {'negative': 'Negative', 'positive': 'Positive', 'neutral': 'Neutral'}
input_ec = open('Tfd.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('svc.pkl', 'rb')
model = pkl.load(input_md)

st.title('Sentiment Analysis\' Feedback')
image_path_or_url = 'https://strategyeducation.co.uk/wp-content/uploads/2020/05/The-Power-of-Feedback-Which-When-and-How.jpg'
response = requests.get(image_path_or_url)
image = Image.open(BytesIO(response.content))
st.image(image, use_column_width=True)
st.header('Write a feedback')
text = st.text_area('', '')

if text != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([text])
    rs = str((model.predict(feature_vector))[0])
    st.header('Result')
    st.write(class_list[rs])
