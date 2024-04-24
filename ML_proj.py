# 1. Setup and Introduction
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 2. Data Handling
def load_data(dataset_name):
    if dataset_name == 'IRIS':
        data = datasets.load_iris()
    elif dataset_name == 'Digits':
        data = datasets.load_digits()
    else:
        data = None
    return data

# 3. Model Selection
def get_classifier(classifier_name):
    if classifier_name == 'Logistic Regression':
        return LogisticRegression()
    elif classifier_name == 'Neural Networks':
        return MLPClassifier(max_iter=1000)
    elif classifier_name == 'Naïve Bayes':
        return GaussianNB()
    else:
        return None

# 4. User Input
def get_user_input(data):
    inputs = []
    for i in range(data.data.shape[1]):
        inputs.append(st.number_input(f'Enter value for {data.feature_names[i]}'))
    return inputs

# 5. Prediction
def make_prediction(model, inputs):
    prediction = model.predict([inputs])
    return prediction

# 6. User Interface
def main():
    st.title("Machine Learning Model Predictor")
    dataset_name = st.sidebar.selectbox("Select Dataset", ('IRIS', 'Digits'))
    data = load_data(dataset_name)

    if data:
        st.write(f"Selected Dataset: {dataset_name}")
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

        classifier_name = st.sidebar.selectbox("Select Classifier", ('Logistic Regression', 'Neural Networks', 'Naïve Bayes'))
        model = get_classifier(classifier_name)
        if model:
            st.write(f"Selected Classifier: {classifier_name}")

            if st.button("Train Model"):
                model.fit(X_train, y_train)
                st.write("Model trained successfully!")

            if st.sidebar.checkbox("Show Accuracy"):
                if classifier_name != 'Neural Networks':
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Accuracy: {accuracy}")

            if st.sidebar.checkbox("Make Prediction"):
                inputs = get_user_input(data)
                prediction = make_prediction(model, inputs)
                st.write("Prediction:", prediction)

# 7. Documentation
if __name__ == "__main__":
    main()
