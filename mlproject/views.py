# Import necessary tools and libraries.
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, r2_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Load diamond data from a file on the computer.
df = pd.read_csv('C:\\Users\\Amo\\Desktop\\mlp\\diamonds.csv')

# Convert text data like 'cut', 'color', and 'clarity' into a number format.
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(df[['cut', 'color', 'clarity']]).toarray()
encoded_cols = encoder.get_feature_names_out(['cut', 'color', 'clarity'])
df_encoded = pd.concat([df, pd.DataFrame(encoded_features, columns=encoded_cols)], axis=1)
df_encoded = df_encoded.drop(columns=['cut', 'color', 'clarity'])

# Split the data into features (X) and what we want to predict (y, which is the price).
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Group diamond prices into categories: low, medium, or high.
def discretize_prices(values):
    bins = [0, 1000, 5000, np.inf]
    labels = ['low', 'medium', 'high']
    return pd.cut(values, bins=bins, labels=labels)

# Convert the real prices in the train and test sets into these categories.
y_train_discrete = discretize_prices(y_train)
y_test_discrete = discretize_prices(y_test)

# Train a model using the Support Vector Machine method.
svm_model = SVC()
svm_model.fit(X_train, y_train_discrete)
svm_predictions = svm_model.predict(X_test)

# Train a model using the Naive Bayes method.
nb_model = GaussianNB()
nb_model.fit(X_train, y_train_discrete)
nb_predictions = nb_model.predict(X_test)

# Train a model using the Decision Tree method.
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train_discrete)
dt_predictions = dt_model.predict(X_test)

# Train a model using the Random Forest method.
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train_discrete)
rf_predictions = rf_model.predict(X_test)

# List all the models and their predictions.
models = ['Support Vector Machine', 'Naive Bayes', 'Decision Tree', 'Random Forest']
predictions = [svm_predictions, nb_predictions, dt_predictions, rf_predictions]

# Convert labels to numbers for r2 calculation.
def labels_to_numbers(labels):
    mapping = {'low': 0, 'medium': 1, 'high': 2}
    return [mapping[label] for label in labels]

# Show the results for a chosen prediction method.
def display_results(request, model_name=None):
    if model_name:
        index = models.index(model_name)
        pred = predictions[index]
        
        y_test_numeric = labels_to_numbers(y_test_discrete)
        pred_numeric = labels_to_numbers(pred)
        
        accuracy = accuracy_score(y_test_discrete, pred)
        precision = precision_score(y_test_discrete, pred, average='weighted', labels=np.unique(pred))
        r2 = r2_score(y_test_numeric, pred_numeric)

        context = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'r2': r2,
        }
        return render(request, 'model_results.html', context)
    else:
        return render(request, 'select_model.html')

# Page to select which prediction method to view.
def select_model(request):
    return render(request, 'select_model.html')

# Homepage for the website.
def homepage(request):
    return render(request, 'homepage.html')
