import streamlit as st
import pickle as pkl
import numpy as np
import sklearn

class_list = {'0': 'Negative', '1': 'Positive', '2': 'Neutral'}
input_ec = open('ec_vsfc.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vsfc.pkl', 'rb')
model = pkl.load(input_md)

st.title('Sentiment Analysis from Vietnamese Students\' Feedback')

st.header('Write a feedback')
text = st.text_area('', '')

if text != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([text])
    rs = str((model.predict(feature_vector))[0])
    st.header('Result')
    st.write(class_list[rs])
