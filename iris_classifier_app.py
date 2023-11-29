
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')


st.title('Iris Dataset Classifier App')


selected_features = st.sidebar.multiselect('Select Features:', X.columns)


if selected_features:
    X = X[selected_features]


classifier_name = st.sidebar.selectbox('Select Classifier:', ['Random Forest', 'SVM'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if classifier_name == 'Random Forest':
    classifier = RandomForestClassifier()
else:
    st.warning("Support Vector Machine (SVM) classifier is not implemented in this example.")
    st.stop()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
st.subheader(f'Accuracy: {accuracy:.2f}')


st.subheader('Confusion Matrix:')
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
st.table(confusion_matrix)


if classifier_name == 'Random Forest':
    st.subheader('Feature Importance:')
    feature_importance = pd.Series(classifier.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    st.pyplot()

