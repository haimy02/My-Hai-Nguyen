from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle as pickle

iris = load_iris()
X,y = iris.data, iris.target

# # Displaying basic info about the dataset
# st.write('## Iris Dataset')
# st.write(iris.data)
# st.write('Shape of dataset:', iris.data.shape)

print(X)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)

clf=RandomForestClassifier()
clf.fit(X_train, y_train)

print(clf.score(X_test,y_test))
print("Saving model to pickle file")
pickle.dump(clf, open("iris_model1.pkl",'wb'))

# prediction = clf.predict(iris.data) 

# import streamlit as st

# st.subheader('Prediction')
# st.write(iris.target_names[prediction])


import streamlit as st
import pickle as pickle
from sklearn.datasets import load_iris

iris = load_iris()

model = pickle.load(open("iris_model.pkl1", "rb"))

st.title('Iris Classifier')

#Sidebar for user input
st.sidebar.title('Input Parameters')
sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)

#Make predictions
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

st.write('## Prediction: ')
st.write(iris.target_names[prediction[0]])
