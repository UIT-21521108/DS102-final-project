import streamlit as st
import pickle as pkl
import numpy as np
import sklearn

class_list = {'negative': 'Negative', 'positive': 'Positive', 'neutral': 'Neutral'}
input_ec = open('Tfd.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('svc.pkl', 'rb')
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
