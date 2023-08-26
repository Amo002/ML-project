# MLP Django App

The `mlp` Django app is designed for training and predicting diamond prices based on their attributes using various machine learning models. This app provides a UI for selecting a prediction model from a set of models, and then displaying the performance metrics of the chosen model on the test dataset.

## Features
- Load diamond data from a CSV file.
- Preprocess the data to transform text features into numeric format using OneHotEncoding.
- Train various models on the dataset, including:
  - Support Vector Machine (SVM)
  - Naive Bayes (GaussianNB)
  - Decision Tree
  - Random Forest
- Display model performance metrics such as accuracy, precision, and R-squared value.

## Structure
- **views.py**: Contains the core logic for data processing, training the models, and rendering views.
- **urls.py**: Handles the routing for the application.
- **templates**: Houses the HTML templates (`model_results.html` and `select_model.html`) for rendering the views.

## Usage

1. **Homepage**: Open the homepage which will lead to the model selection page.
2. **Model Selection**: Select a prediction method from the given models.
3. **Results**: After selecting a model, you will be presented with its performance metrics.

## Setup
1. Ensure you have Django installed.
2. Navigate to the main directory `mlp`.
3. Run the command `python manage.py runserver` to start the server.
4. Open a web browser and go to `http://localhost:8000/` to access the app.

## Dependencies
- Django
- scikit-learn
- pandas
- numpy

## Notes
- The dataset is loaded from the path `C:\Users\Amo\Desktop\mlp\diamonds.csv`. Make sure the file exists or update the path accordingly.
- For best performance and more accurate predictions, consider using a larger dataset or tweaking the model parameters.

## Contribution
Feel free to contribute to this project by creating a pull request or raising an issue for any bugs or enhancements.

## License
This project is licensed under the MIT License.
