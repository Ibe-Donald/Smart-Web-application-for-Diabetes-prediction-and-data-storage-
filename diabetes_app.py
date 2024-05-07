#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mysql.connector

# In[53]:



# Establish a connection to the MySQL database
mydb = mysql.connector.connect(
  host= 'enter hostname',
  user='enter user name',
  password='enter password',
  database='enter database name'
)

# Create a cursor object to execute SQL queries
mycursor = mydb.cursor()


# Create a table to store user inputs
create_table_query = """
CREATE TABLE IF NOT EXISTS user_inputs (
  pregnancies INT,
  glucose FLOAT,
  blood_pressure FLOAT,
  skin_thickness FLOAT,
  insulin FLOAT,
  bmi FLOAT,
  dpf FLOAT,
  age INT,
  prediction INT
)
"""
mycursor.execute(create_table_query)


# In[54]:


import streamlit as st


# In[55]:


# In[56]:


df = pd.read_csv(r'C:\\Users\\Use\\Desktop\\newdiabetes.csv')


# In[57]:


df[['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin','BMI']] = df[['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin','BMI']].replace(0, np.NaN)

# In[58]:


df['Pregnancies'].fillna(df['Pregnancies'].median(), inplace = True)
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Glucose'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)


# In[59]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range= (0,1))

new_df = scaler.fit_transform(df)


# In[60]:


dataset_scaled = pd.DataFrame(new_df)


# In[61]:


X = dataset_scaled.iloc[:, [0,1,2,3,4,5,6,7]].values
Y = dataset_scaled.iloc[:, 8].values


# In[62]:




# In[63]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)


# In[64]:


from sklearn.ensemble import RandomForestClassifier


# In[65]:


rf_model = RandomForestClassifier( n_estimators = 50, min_samples_split = 2, min_samples_leaf = 2, max_features = 'log2', max_depth = None, class_weight ='balanced', bootstrap = False)


# In[66]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size = 0.20, random_state=42, stratify=Y_resampled)


# In[67]:


rf_model.fit(X_train, Y_train)


# In[68]:


# Streamlit app
st.title('Diabetes Prediction App')
st.title('Please note your data entered will be collected')

# Get user inputs
pregnancies = st.number_input('Number of times pregnant', min_value=0, step=1)
glucose = st.number_input('Plasma glucose concentration (mg/dL)', min_value=0.0)
blood_pressure = st.number_input('Diastolic blood pressure (mm Hg)', min_value=0.0)
skin_thickness = st.number_input('Triceps skin fold thickness (mm)', min_value=0.0)
insulin = st.number_input('2-Hour serum insulin (mu U/ml)', min_value=0.0)
bmi = st.number_input('Body mass index (weight in kg/(height in m)^2)', min_value=0.0)
dpf = st.number_input('Diabetes pedigree function', min_value=0.0)
age = st.number_input('Age (years)', min_value=0)

sumbit = st.button('Submit')


if sumbit:
    # Create a sample input
    sample_input = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

    # Make the prediction
    prediction = rf_model.predict(sample_input)

    # Display the result
    if prediction[0] == 0:

        st.success('The person is not likely to have diabetes.')

    else:
        st.warning('The person is likely to have diabetes.')


    # Insert the user inputs and prediction into the MySQL table
    insert_query = """
    INSERT INTO user_inputs (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, prediction)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, prediction[0])
    mycursor.execute(insert_query, values)
    mydb.commit()


# In[ ]:




