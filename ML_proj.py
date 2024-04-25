# import streamlit as st
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# # Load datasets
# iris = datasets.load_iris()
# digits = datasets.load_digits()

# # Function to train the model
# def train_model(classifier_name, X_train, y_train):
#     if classifier_name == 'Logistic Regression':
#         model = LogisticRegression()
#         model.fit(X_train, y_train)
#         return model
#     elif classifier_name == 'Neural Networks':
#         model = MLPClassifier(max_iter=1000)
#         model.fit(X_train, y_train)
#         return model
#     elif classifier_name == 'Naïve Bayes':
#         model = GaussianNB()
#         model.fit(X_train, y_train)
#         return model
#     else:
#         model = None

#     # if model:
#     #     model.fit(X_train, y_train)
#     #     return model
#     # else:
#     #     return None

# # Function to make predictions
# def make_prediction(model, inputs):
#     prediction = model.predict(inputs)
#     accuracy = accuracy_score(y_test,y_train)
#     return accuracy

# def main():
#     st.title("Machine Learning Model Predictor")

#     # Sidebar options
#     dataset_name = st.sidebar.selectbox("Select Dataset", ('IRIS', 'Digits'))
#     classifier_name = st.sidebar.selectbox("Select Classifier", ('Logistic Regression', 'Neural Networks', 'Naïve Bayes'))

#     # Load selected dataset
#     if dataset_name == 'IRIS':
#         data = iris
#     else:
#         data = digits

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

#     # Train the model
#     if st.button("Train Model"):
#         model = train_model(classifier_name, X_train, y_train)
#         if model:
#             st.session_state.model = model
#             st.write("Model trained successfully!")
#         else:
#             st.write("Error: Model not trained.")

#     # Make predictions
#     if st.button("Make Prediction"):
#         if 'model' not in st.session_state:
#             #st.write(model.type)
#             st.write("Error: Model not trained. Please train the model first.")
#         else:
#             model= st.session_state.model 
#             inputs = st.text_input("Enter feature values separated by commas (e.g., 5.1, 3.5, 1.4, 0.2):")
#             if inputs:
#                 inputs = [[float(x.strip()) for x in inputs.split(',')]]
#                 prediction = make_prediction(model, inputs)
#                 st.write("Prediction:", prediction)

# # Run the app
# if __name__ == "__main__":
#     main()


import streamlit as st
from sklearn.datasets import load_iris, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1. Setup and Introduction
st.title("Machine Learning Model Predictor")
st.write("Welcome to the Machine Learning Model Predictor! Select your dataset and algorithm to get started.")

# 2. Data Handling
datasets = {
    "IRIS": load_iris(),
    "Digits": load_digits()
}
selected_dataset = st.sidebar.selectbox("Select Dataset", list(datasets.keys()))

# 3. Model Selection
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Naïve Bayes": GaussianNB(),
    "Neural Networks": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
}
selected_classifier = st.sidebar.selectbox("Select Classifier", list(classifiers.keys()))

# 4. User Input
st.write("### Enter Feature Values")
if selected_dataset == "IRIS":
    feature_names = datasets["IRIS"].feature_names
    user_inputs = []
    for feature in feature_names:
        value = st.number_input(f"Enter value for {feature}", step=0.01)
        user_inputs.append(value)
else:
    #feature_names = [f"Pixel {i}" for i in range(64)]  # Digits dataset has 64 features
    st.write("features in digits dataset are taken from testing split")



# 5. Prediction
if st.button("Make Prediction"):
    selected_model = classifiers[selected_classifier]

    if selected_dataset == "IRIS":
        X_train, X_test, y_train, y_test = train_test_split(datasets["IRIS"].data, datasets["IRIS"].target, test_size=0.2, random_state=42)
        selected_model.fit(X_train, y_train)  # Train the model

        X = np.array(user_inputs).reshape(1, -1)  # Reshape inputs to match model's input shape
        prediction = selected_model.predict(X)
        y_pred = selected_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        st.write("### Prediction Result")
        st.write(f"The model predicts the class as: {prediction[0]}")
        st.write(f"The model accuracy is: {accuracy}")
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(datasets["Digits"].data, datasets["Digits"].target, test_size=0.2, random_state=42)
        selected_model.fit(X_train, y_train)
        prediction = selected_model.predict(X_test)
        accuracy = accuracy_score(y_test,prediction)
        st.write("### Prediction Result")
        st.write(f"The model accuracy is: {accuracy}")
        cm = confusion_matrix(y_test,prediction)
        st.write("Confusion Matrix:")
        st.write(cm)   

# 6. User Interface (Already organized in logical structure)

# 7. Documentation (Not included here, but important for further development)
