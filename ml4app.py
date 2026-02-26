import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Title
# -------------------------------
st.title("🌸 KNN Classification - Iris Dataset")
st.write("Machine Learning Practical 4 - Streamlit Output")

# -------------------------------
# Load Dataset
# -------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(X.head())

# -------------------------------
# Sidebar - Select K value
# -------------------------------
k = st.sidebar.slider("Select K value", min_value=1, max_value=15, value=5)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# -------------------------------
# Prediction & Accuracy
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# Plotly Graph (No Matplotlib)
# -------------------------------
st.subheader("Interactive Scatter Plot")

fig = px.scatter(
    X,
    x="sepal length (cm)",
    y="sepal width (cm)",
    color=y.astype(str),
    title="Sepal Length vs Sepal Width",
    labels={"color": "Species"}
)

st.plotly_chart(fig)
