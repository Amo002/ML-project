from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, precision_score
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

# Your data loading and preprocessing code:
df = pd.read_csv('C:\\Users\\Amo\\Desktop\\mlp\\diamonds.csv')

encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(df[['cut', 'color', 'clarity']]).toarray()
encoded_cols = encoder.get_feature_names_out(['cut', 'color', 'clarity'])
df_encoded = pd.concat([df, pd.DataFrame(encoded_features, columns=encoded_cols)], axis=1)
df_encoded = df_encoded.drop(columns=['cut', 'color', 'clarity'])

X = df_encoded.drop(columns=['price'])
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def discretize_prices(values):
    bins = [0, 1000, 5000, np.inf]
    labels = ['low', 'medium', 'high']
    return pd.cut(values, bins=bins, labels=labels)

y_test_discrete = discretize_prices(y_test)

# Support Vector Machine Algorithm
svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

# ID3 Algorithm - Decision Tree
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

models = ['Support Vector Machine', 'Naive Bayes', 'Decision Tree', 'Random Forest']
predictions = [svm_predictions, nb_predictions, dt_predictions, rf_predictions]
predictions_discrete = [discretize_prices(pred) for pred in predictions]

def display_results(request, model_name=None):
    if model_name:
        index = models.index(model_name)
        pred = predictions_discrete[index]
        accuracy = accuracy_score(y_test_discrete, pred)
        precision = precision_score(y_test_discrete, pred, average='weighted', labels=np.unique(pred))
        r2 = r2_score(y_test, predictions[index])

        context = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'r2': r2,
        }
        return render(request, 'model_results.html', context)
    else:
        return render(request, 'select_model.html')

def select_model(request):
    return render(request, 'select_model.html')

def homepage(request):
    return render(request, 'homepage.html')
