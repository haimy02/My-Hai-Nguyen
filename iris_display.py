import pickle as pickle
import streamlit as st
from sklearn.datasets import load_iris

iris = load_iris()

model = pickle.load(open("iris_model.pkl", "rb"))

#Sidebar for user input
st.sidebar.title('Iris Classifier')
sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)

#Make predictions
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

st.write('## Prediction: ')
st.write(iris.target_names[prediction[0]])