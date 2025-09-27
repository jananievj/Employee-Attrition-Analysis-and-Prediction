import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Load saved files
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\Employee-Attrition\preprocessed_employee_attrition.csv")

# Prediction function
def predict_attrition(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Encode categoricals
    for col in encoders:
        if col in input_df.columns:
            le = encoders[col]
            input_df[col] = le.transform([input_df[col].values[0]])[0]
    
    # Feature engineering
    input_df['TenurePerJobLevel'] = input_df['YearsAtCompany'] / (input_df['JobLevel'] + 1)
    input_df['PromotionLag'] = input_df['YearsSinceLastPromotion'] / (input_df['YearsAtCompany'] + 1)
    
    # Ensure column order matches training
    training_columns = ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 
                        'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 
                        'MaritalStatus', 'MonthlyIncome', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 
                        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                        'YearsWithCurrManager', 'TenurePerJobLevel', 'PromotionLag']
    input_df = input_df[training_columns]
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    proba = model.predict_proba(input_scaled)[:, 1][0]
    pred = 1 if proba >= 0.5 else 0
    return pred, proba

# Set page config
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🚓 Employee Attrition Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>An interactive tool for predicting employee turnover using Random Forest and analyzing trends.</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Model Evaluation", "Predict Attrition", "Employee Insights"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("Visualizations of key trends and factors influencing employee attrition.")
    st.subheader("Distribution of Attrition")
fig1, ax1 = plt.subplots()
sns.countplot(x='Attrition', data=df, ax=ax1)
ax1.set_title("Distribution of Attrition")
st.pyplot(fig1)

# Show counts
st.write("Attrition Counts:")
st.write(df['Attrition'].value_counts())

# -----------------------
# Pairplot for selected features
# -----------------------
st.subheader("Pairplot: Age, MonthlyIncome, YearsAtCompany vs Attrition")
# Pairplot cannot be directly used with st.pyplot, need to use fig object
pairplot_fig = sns.pairplot(df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'Attrition']], hue="Attrition", palette="husl")
st.pyplot(pairplot_fig)

# -----------------------
# kdeplot for Density Estimation
# -----------------------
st.subheader("Density Estimation")
# Monthly Income KDE
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.kdeplot(df["MonthlyIncome"], fill=True, color="blue", ax=ax1)
ax1.set_title("KDE Plot of Monthly Income")
st.pyplot(fig1)
# Years at Company KDE
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.kdeplot(df["YearsAtCompany"], fill=True, color="green", ax=ax2)
ax2.set_title("KDE Plot of Years at Company")
st.pyplot(fig2)  

with tab2:
    st.header("Model Evaluation")
    st.markdown("Performance metrics for the Random Forest model.")
    
    st.subheader("Random Forest")
    # Replace with your actual metrics from training output
    st.text("Best Params: {'n_estimators': 200, 'max_depth': None}")  # Example
    st.text("Accuracy: 0.94")  # Example
    st.text("Precision: 0.96")
    st.text("Recall: 0.92")
    st.text("F1 Score: 0.94")
    st.text("ROC AUC: 0.98")
    st.text("Confusion Matrix:\n[[299, 12], [23, 283]]")
    st.text("Classification Report:\n  precision    recall  f1-score\n0       0.93     0.96     0.94\n1       0.96      0.92      0.94")

with tab3:
    st.header("Predict Attrition for a New Employee")
    st.markdown("Enter employee details to predict their likelihood of leaving using the Random Forest model.")
    
    input_data = {}
    
    with st.form("employee_form"):
        col1, col2 = st.columns([1.5, 1.5], gap="large")
        
        with col1:
            input_data['Age'] = st.number_input("Age", min_value=18, max_value=60, value=30)
            input_data['BusinessTravel'] = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
            input_data['Department'] = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
            input_data['DistanceFromHome'] = st.number_input("Distance From Home", min_value=1, max_value=30, value=5)
            input_data['Education'] = st.selectbox("Education", [1, 2, 3, 4, 5])
            input_data['EducationField'] = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
            input_data['EnvironmentSatisfaction'] = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
            input_data['Gender'] = st.selectbox("Gender", ['Male', 'Female'])
            input_data['JobInvolvement'] = st.selectbox("Job Involvement", [1, 2, 3, 4])
            input_data['JobLevel'] = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            input_data['JobRole'] = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
            input_data['JobSatisfaction'] = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            input_data['MaritalStatus'] = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        
        with col2:
            input_data['MonthlyIncome'] = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
            input_data['OverTime'] = st.selectbox("OverTime", ['Yes', 'No'])
            input_data['PercentSalaryHike'] = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=10)
            input_data['PerformanceRating'] = st.selectbox("Performance Rating", [1, 2, 3, 4])
            input_data['RelationshipSatisfaction'] = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
            input_data['StockOptionLevel'] = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            input_data['TotalWorkingYears'] = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
            input_data['TrainingTimesLastYear'] = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
            input_data['WorkLifeBalance'] = st.selectbox("Work Life Balance", [1, 2, 3, 4])
            input_data['YearsAtCompany'] = st.number_input("Years At Company", min_value=0, max_value=40, value=3)
            input_data['YearsInCurrentRole'] = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
            input_data['YearsSinceLastPromotion'] = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
            input_data['YearsWithCurrManager'] = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        pred, proba = predict_attrition(input_data)
        if pred == 1:
            st.error(f"The employee is at high risk of attrition (probability: {proba:.2%}).So the Employee is likely to Leave.")
        else:
            st.success(f"The employee is likely to stay (probability of staying: {1 - proba:.2%}).")
    
    st.markdown("### Dataset Preview")
    st.dataframe(df.head())
with tab4:
   # Predict attrition risk
    def predict_attrition(input_data):
        input_df = pd.DataFrame([input_data])
   # Create 3 side-by-side columns
    col1, col2, col3 = st.columns(3)

with col1:
    high_risk = df.nlargest(15, 'Attrition')
    high_risk['JobRole'] = high_risk['JobRole'].map(
        dict(zip(encoders['JobRole'].transform(encoders['JobRole'].classes_),
                 encoders['JobRole'].classes_))
    )
    st.subheader("High-Risk Employees")
    st.dataframe(high_risk[['JobRole', 'Attrition', 'PerformanceRating']], height=300)

with col2:
    high_satisfaction = df.nlargest(10, 'JobSatisfaction')
    st.subheader("High Job Satisfaction")
    st.dataframe(high_satisfaction[['Age', 'JobSatisfaction', 'Attrition']], height=300)

with col3:
    high_performance = df.nlargest(10, 'PerformanceRating')
    high_performance['Department'] = high_performance['Department'].map(
        dict(zip(encoders['Department'].transform(encoders['Department'].classes_),
                 encoders['Department'].classes_))
    )
    st.subheader("High Performance")
    st.dataframe(high_performance[['Department', 'PerformanceRating', 'JobSatisfaction']], height=300)

# Footer
st.markdown("---")
st.markdown("Developed by JANANI ELANGO | Powered by Streamlit")