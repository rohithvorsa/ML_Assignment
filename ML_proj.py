import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

# Function to train the model
def train_model(classifier_name, X_train, y_train):
    if classifier_name == 'Logistic Regression':
        model = LogisticRegression()
    elif classifier_name == 'Neural Networks':
        model = MLPClassifier(max_iter=1000)
    elif classifier_name == 'Naïve Bayes':
        model = GaussianNB()
    else:
        model = None

    if model:
        model.fit(X_train, y_train)
        return model
    else:
        return None

# Function to make predictions
def make_prediction(model, inputs):
    prediction = model.predict(inputs)
    return prediction

def main():
    st.title("Machine Learning Model Predictor")

    # Sidebar options
    dataset_name = st.sidebar.selectbox("Select Dataset", ('IRIS', 'Digits'))
    classifier_name = st.sidebar.selectbox("Select Classifier", ('Logistic Regression', 'Neural Networks', 'Naïve Bayes'))

    # Load selected dataset
    if dataset_name == 'IRIS':
        data = iris
    else:
        data = digits

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Train the model
    if st.button("Train Model"):
        model = train_model(classifier_name, X_train, y_train)
        if model:
            st.write("Model trained successfully!")
        else:
            st.write("Error: Model not trained.")

    # Make predictions
    if st.button("Make Prediction"):
        if 'model' not in st.session_state:
            st.write("Error: Model not trained. Please train the model first.")
        else:
            model = st.session_state.model
            inputs = st.text_input("Enter feature values separated by commas (e.g., 5.1, 3.5, 1.4, 0.2):")
            if inputs:
                inputs = [[float(x.strip()) for x in inputs.split(',')]]
                prediction = make_prediction(model, inputs)
                st.write("Prediction:", prediction)

# Run the app
if __name__ == "__main__":
    main()
