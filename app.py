import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('DataScience_salaries_2024.csv')

# Data Cleaning and Preprocessing
# Fill missing values in numerical columns with their mean
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Encoding categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split the data into features (X) and target variable (y)
X = df.drop('salary_in_usd', axis=1)
y = df['salary_in_usd']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training and Evaluation

# Random Forest Model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_y_pred)
lr_mae = mean_absolute_error(y_test, lr_y_pred)

# Print model evaluation metrics with R² score as a percentage
print("Random Forest Model Evaluation:")
print("R² Score (Accuracy):", f"{round(rf_r2 * 100, 2)}%")
print("Mean Absolute Error:", round(rf_mae, 2))

print("\nLinear Regression Model Evaluation:")
print("R² Score (Accuracy):", f"{round(lr_r2 * 100, 2)}%")
print("Mean Absolute Error:", round(lr_mae, 2))

# Streamlit code for the salary prediction app
st.title("Data Science Salary Predictor")

# Create input fields for the app
experience_level = st.selectbox("Experience Level", ["Entry Level", "Junior", "Mid-Senior", "Senior"])
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract"])

# Options for job titles and locations based on possible dataset values
job_titles = ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "Data Engineer", "Research Scientist"]
locations = ["San Francisco", "New York", "Austin", "Seattle", "Remote"]

job_title = st.selectbox("Job Title", job_titles)
location = st.selectbox("Location", locations)
company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

# Prediction function
def predict_salary(experience_level, employment_type, job_title, location, company_size, model):
    # Create DataFrame with input values
    input_data = pd.DataFrame([[experience_level, employment_type, job_title, location, company_size]], 
                              columns=['experience_level', 'employment_type', 'job_title', 'location', 'company_size'])
    
    # Convert categorical variables to dummy variables
    input_data = pd.get_dummies(input_data)

    # Align input data columns with training data
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_data = scaler.transform(input_data)  # scale the input data

    # Make prediction
    predicted_salary = model.predict(input_data)
    
    return predicted_salary[0]

# Model selection in the app (not displayed in output)
model_choice = st.selectbox("Choose the model", ["Random Forest", "Linear Regression"])

# Make prediction when the button is clicked
if st.button("Predict Salary"):
    selected_model = rf_model if model_choice == "Random Forest" else lr_model
    predicted_salary = predict_salary(experience_level, employment_type, job_title, location, company_size, selected_model)
    # Convert annual salary to monthly
    monthly_salary = predicted_salary / 12
    st.write("Predicted Annual Salary: $", round(predicted_salary, 2))
    st.write("Predicted Monthly Salary: $", round(monthly_salary, 2))
