Project Overview: Employee Attrition Prediction
1. Objective

The goal of this project is to predict employee attrition (whether an employee is likely to leave) using historical employee data and machine learning. Additionally, it provides an interactive dashboard for HR analytics to visualize attrition trends, high-risk employees, and other workforce metrics.

2. Dataset

Source: Employee dataset (Employee-Attrition.csv and preprocessed_employee_attrition.csv)

Features included demographic, job, and performance-related attributes such as:

Numeric features: Age, MonthlyIncome, YearsAtCompany, YearsInCurrentRole, etc.

Categorical features: JobRole, Department, BusinessTravel, MaritalStatus, Gender, OverTime, etc.

Target variable: Attrition (Yes/No or 1/0)

3. Data Preprocessing

Dropped unnecessary columns like EmployeeCount, Over18, StandardHours, etc. for modeling.

Categorical encoding: Used LabelEncoder for features like JobRole, Department, OverTime, etc.

Feature scaling: Applied StandardScaler to numeric features to normalize the data.

Handling imbalanced data: Used RandomOverSampler to balance the target class (Attrition) to avoid bias toward non-attrition employees.

4. Model Training

You trained two models for attrition prediction:

a) Logistic Regression

Used as a baseline.

Metrics calculated: Accuracy, Recall, Precision, F1-score, AUROC.

b) Random Forest Classifier

Main predictive model.

Calculated feature importances to see which factors influence attrition the most.

Metrics calculated: Accuracy, Precision, Recall, F1-score, ROC AUC.

Saved trained model using pickle.

5. Feature Engineering

Created new features to improve model performance:

TenurePerJobLevel: Years at company per job level.

PromotionLag: Years since last promotion relative to tenure.

6. Streamlit Dashboard

You built an interactive web dashboard to showcase insights:

a) Tabs:

EDA (Exploratory Data Analysis)

Distribution of attrition.

Pairplots for Age, MonthlyIncome, YearsAtCompany vs Attrition.

Boxplots of numeric features.

KDE plots for continuous variables.

Model Evaluation

Showed model metrics like accuracy, precision, recall, F1-score, ROC AUC.

Confusion matrix and classification report.

Predict Attrition

Form for entering employee details.

Encodes categorical features, scales numeric ones, and predicts attrition probability.

Highlights high-risk employees, high job satisfaction, and high performance employees side-by-side.

Shows employee attrition risk in percentage.

7. Special Features Added

High-risk, high job satisfaction, and high performance tables displayed side by side using st.columns.

Original categorical names shown instead of numeric codes using encoders.

KDE and other plots converted to Streamlit-compatible format using st.pyplot(fig).

8. Outputs

Interactive prediction of attrition probability for new employees.

Dashboard provides actionable insights for HR:

Who is at high risk of leaving.

Employees with high satisfaction and performance.

Overall trends and correlations in the workforce data.

9. Next Steps / Recommendations

Deploy the Streamlit dashboard on a server for HR access.

Regularly update the model with new employee data for accuracy.

Incorporate more features like engagement scores, training participation, or survey results for better prediction.
