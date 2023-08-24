Django ML Project
Description
This Django project is designed to predict diamond prices using various machine learning models. The project uses the diamonds.csv dataset and implements four different models: Support Vector Machine, Naive Bayes, Decision Tree, and Random Forest. The results of the predictions are displayed on a web interface, allowing users to select a model and view its performance metrics.

Features
Data Loading and Preprocessing: The dataset is loaded and preprocessed using pandas and scikit-learn. Features are one-hot encoded, and the data is split into training and testing sets.

Machine Learning Models: The project implements four models:

Support Vector Machine
Naive Bayes
Decision Tree
Random Forest
Web Interface: The project provides a web interface where users can:

Select a machine learning model
View the results, including accuracy, precision, and R^2 score
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/AmoAlberqdar/ML-project.git
Navigate to the project directory:

bash
Copy code
cd ML-project
Install the required packages:

Copy code
pip install django scikit-learn pandas numpy
Run the Django development server:

Copy code
python manage.py runserver
Open your browser and navigate to http://127.0.0.1:8000/ to access the web interface.

Usage
On the homepage, click on the link to select a model.
Choose a machine learning model to view its results.
The results page will display the performance metrics for the selected model.
Dependencies
Django
scikit-learn
pandas
numpy
Contribution
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.

License
This project is open-source and available under the MIT License.
