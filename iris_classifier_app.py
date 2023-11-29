# iris_classifier_app.py
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Streamlit app
st.title('Iris Dataset Classifier App')

# Sidebar - Select Features
selected_features = st.sidebar.multiselect('Select Features:', X.columns)

# Filter dataset based on selected features
if selected_features:
    X = X[selected_features]

# Sidebar - Select Classifier
classifier_name = st.sidebar.selectbox('Select Classifier:', ['Random Forest', 'SVM'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
if classifier_name == 'Random Forest':
    classifier = RandomForestClassifier()
else:
    st.warning("Support Vector Machine (SVM) classifier is not implemented in this example.")
    st.stop()

classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f'Accuracy: {accuracy:.2f}')

# Display confusion matrix
st.subheader('Confusion Matrix:')
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
st.table(confusion_matrix)

# Visualize feature importance for Random Forest
if classifier_name == 'Random Forest':
    st.subheader('Feature Importance:')
    feature_importance = pd.Series(classifier.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    st.pyplot()

