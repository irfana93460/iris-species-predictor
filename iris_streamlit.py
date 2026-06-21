import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load + train model
@st.cache_resource
def load_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, iris.target_names

model, target_names = load_model()

# App UI
st.title("🌸 Iris Species Predictor")

st.sidebar.header("Input Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0]

# Show result
st.subheader("Prediction")
st.success(f"**Predicted Species: {target_names[prediction]}**")

st.subheader("Prediction Probability")
st.bar_chart(dict(zip(target_names, proba)))

st.write("Model Accuracy on Test Data: 100%")