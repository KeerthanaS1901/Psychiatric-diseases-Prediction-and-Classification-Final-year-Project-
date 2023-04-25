import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('DETECTION AND CLASSIFICATION OF PYSCHIATRIAC DISEASES')

st.write("""
TEAM MEMBERS:
""")

st.write("""
RENUKANANDA T D


""")

st.write("""
KEERTHANA S


""")

st.write("""
VIDYASHREE K S

""")

st.write("""
THRUPTHI M S

""")

st.title("RNN CLASSIFIER")
image= 'rnn_piechart.png'
st.image(image, caption='PIE CHART FOR RNN CLASSIFIER',use_column_width=True)
st.write("The accuracy obtained for RNN CLASSIFIER is aprroximately 58%")

st.title("GRU CLASSIFIER")
image= 'gru_piechart.png'
st.image(image, caption='PIE CHART FOR GRU CLASSIFIER',use_column_width=True)
st.write("The accuracy obtained for GRU CLASSIFIER is aprroximately 65.5")

st.title("LSTM CLASSIFIER")
image= 'lstm_piechart.png'
st.image(image, caption='PIE CHART FOR LSTM CLASSIFIER',use_column_width=True)
st.write("The accuracy obtained for LSTM CLASSIFIER is aprroximately 72%")


