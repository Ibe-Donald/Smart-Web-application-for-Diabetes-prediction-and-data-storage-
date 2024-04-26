# Smart-Web-application-for-Diabetes-prediction-and-data-storage-
---
This diabetes dataset was collected from 2000 people at the Frankfurt Hospital, Germany. There are eight features in the dataset. Among the 2000 samples, 684 people are Diabetes patients, and the rest of them are normal.

Creating a machine learning model from the dataset typically involved several stages, which I will discuss in detail.

1. Data Collection and Preprocessing:
   - The first step was to obtain the Frankfurt Hospital diabetes dataset, which involved accessing it from an online repository 
   - The dataset contained various features or attributes related to the patients, such as age, body mass index (BMI), blood glucose levels, pregnancies, blood pressure, skin thickness, insulin, diabetes pedigree function.
   - Data preprocessing was crucial to handle missing values, outliers, and any inconsistencies in the data. This involved techniques like imputation, normalization, or encoding categorical variables.

2. Exploratory Data Analysis (EDA):
   - EDA was an essential step that allowed understanding the characteristics of the dataset and gaining insights into the relationships between different features and the target variable (diabetes diagnosis).
   - During EDA, the data was visualized using various plots and charts, such as histograms, scatter plots, and correlation matrices, to identify patterns, trends, and potential outliers.
   - EDA also helped identify any potential biases or imbalances in the dataset, which needed to be addressed before building the machine learning model.

3. Feature Selection and Engineering:
   - Feature selection involved identifying the most relevant features or attributes that were likely to have a significant impact on predicting the target variable.
   - This was done using techniques like correlation analysis.

4. Model Selection and Training:
   - Based on the nature of the problem (classification or regression), classification algorithms are used to create the model.
   - Algorithms used for the diabetes prediction included k nearest neighbor, decision trees, random forests, support vector machines, and Xgboost.

5. Model Evaluation and Tuning:
   - After training the model, it was essential to evaluate its performance using appropriate metrics, such as accuracy, precision, recall, F1-score, confusion matrix for classification task   
   - Depending on the evaluation results, the model's hyperparameters needed to be tuned to improve its performance.
   - Techniques like grid search or random search were used for hyperparameter tuning.
   -  The algorithm with the highest performance was selected to be deployed to the web application.

6. Model Deployment:
   - The model created is then deployed to a web application using streamlit. This web application accepts inputs and with this input predicts the likelihood if the patient has diabetes or not
