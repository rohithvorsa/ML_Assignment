# Importing the necessary packages
import streamlit as st
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Introduction
st.title("Machine Learning Model Predictor")
st.write("Welcome to the Machine Learning Model Predictor! Select your dataset and algorithm to get started.")

# Data Handling
datasets = {
    "IRIS": load_iris(),
    "Digits": load_digits()
}
selected_dataset = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))

# Model Selection
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Naïve Bayes": GaussianNB(),
    "Neural Networks": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
}
selected_classifier = st.sidebar.selectbox("Select Classifier", list(classifiers.keys()))

# Taking User Input for the IRIS datset with features like petal length, width; sepal length, width
if selected_dataset == "IRIS":
    st.write("### Enter Feature Values")
    feature_names = datasets["IRIS"].feature_names
    user_inputs = []
    for feature in feature_names:
        value = st.number_input(f"Enter value for {feature}", step=0.01)
        user_inputs.append(value)
# Taking the testing values from testing split of the dataset
else:
    st.write("Features in digits dataset are taken from testing split because manual entry for images is not possible in the case of Digits dataset.")



# designing prediction button
if st.button("Make Prediction"):
    selected_model = classifiers[selected_classifier]
    # If IRIS is selected
    if selected_dataset == "IRIS":
        # training and testing splits from the dataset
        X_train, X_test, y_train, y_test = train_test_split(datasets["IRIS"].data, datasets["IRIS"].target, test_size=0.2, random_state=42)
        selected_model.fit(X_train, y_train)  # Train the model
        #taking the user inputs of lengths and widths of petal and sepal
        X = np.array(user_inputs).reshape(1, -1)  # Reshape inputs to match model's input shape
        # prediction of user input
        prediction = selected_model.predict(X)
        st.write("### Prediction Result")
        st.write(f"The model predicts the class as: {prediction[0]}")

        #Displaying the model accuracy after training the model with training data
        y_pred = selected_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        st.write(f"The model accuracy is: {accuracy}")

    # If the selected model is Digits        
    else:
        #testing and training splits 
        X_train, X_test, y_train, y_test = train_test_split(datasets["Digits"].data, datasets["Digits"].target, test_size=0.2, random_state=42)
        selected_model.fit(X_train, y_train)
        prediction = selected_model.predict(X_test)
        accuracy = accuracy_score(y_test,prediction)
        st.write("### Prediction Result")
        st.write(f"The model accuracy is: {accuracy}")
        cm = confusion_matrix(y_test,prediction)
        # displaying the confusion matrix of the predicted models
        st.write("Confusion Matrix:")
        st.write(cm)   

