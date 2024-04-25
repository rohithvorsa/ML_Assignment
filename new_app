import streamlit as st
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
iris = load_iris()
digits = load_digits()

# Sidebar for selecting dataset and algorithm
st.sidebar.title("Select Dataset and Algorithm")
dataset = st.sidebar.selectbox("Select Dataset", ("Iris", "Digits"))
algorithm = st.sidebar.selectbox("Select Algorithm", ("Logistic Regression", "Naive Bayes", "Neural Network"))

# Display basic dataset information
if dataset == "Iris":
    data = iris
    st.write("### Iris Dataset")
    st.write(data.DESCR)
else:
    data = digits
    st.write("### Digits Dataset")
    st.write(data.DESCR)

# Model training and evaluation
if st.button("Train and Evaluate Model"):
    if dataset == "Iris":
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    if algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "Naive Bayes":
        model = GaussianNB()
    else:
        model = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
