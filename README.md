# Credit Card Fraud Detection

# Overview
This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset used is the creditcard.csv file, which contains credit card transactions made by European cardholders in September 2013.

# Dataset
The dataset consists of 31 columns:
Time: The number of seconds elapsed between this transaction and the first transaction in the dataset.
V1 to V28: The result of a PCA transformation of the original features (due to confidentiality issues, the original features are not provided).
Amount: The transaction amount.
Class: The label for the transaction (0 for non-fraudulent, 1 for fraudulent).
The dataset is highly imbalanced, with the majority of transactions being non-fraudulent.

# Project Structure
The project is structured as follows:
1. Data Preprocessing:
2. Feature Selection:
3. Model Training and Evaluation:
4. Model Performance:

Detailed Steps
1. Data Preprocessing
Load the creditcard.csv file using Pandas.
Inspect the dataset for missing values and handle them appropriately.
Drop rows with missing values as the dataset is large enough to handle this without losing significant information.
Visualize the class distribution using Seaborn to understand the imbalance in the dataset.
2. Feature Selection
Use Recursive Feature Elimination (RFE) with a Logistic Regression estimator to select the top 10 most important features from the dataset.
3. Model Training and Evaluation
Split the data into training and testing sets using StratifiedKFold to maintain the proportion of fraud and non-fraud cases in each fold.
Train various machine learning models including:
Logistic Regression
Nearest Centroid
Artificial Neural Network (ANN)
Evaluate the models using metrics such as accuracy, precision, recall, F1-score, and confusion matrix to determine their performance.
4. Model Performance
Evaluate the performance of the models on the test set.
Use LazyPredict to benchmark multiple models and compare their performance.
Fine-tune the models using GridSearchCV to find the best hyperparameters for the selected models.
Use classification reports and confusion matrices to analyze the performance of the models.

# Results
The Nearest Centroid model achieved high accuracy with a significant imbalance between precision and recall for the fraudulent class.
The Logistic Regression model showed similar results with slightly lower performance in recall for the fraudulent class.
The Artificial Neural Network (ANN) model provided a balanced performance with improvements in recall for the fraudulent class.

# Conclusion
This project demonstrates the process of building and evaluating machine learning models for credit card fraud detection. The results show that it is possible to achieve high accuracy in detecting fraudulent transactions, although challenges remain in balancing precision and recall for the minority class.
