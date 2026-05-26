import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Title
st.title("Loan Approval Predictor")

# Load Dataset
data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Display Dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Handle Missing Values
# Fill numeric missing values with mean
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

num_imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

# Fill categorical missing values with most frequent value
categorical_cols = data.select_dtypes(include=['object']).columns

cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Encode Categorical Columns
le = LabelEncoder()

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Features and Target
X = data.drop(["Loan_ID", "Loan_Status"], axis=1)
st.write(X.columns)
y = data["Loan_Status"]
print(X.columns)
print(len(X.columns))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction Section
st.subheader("Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [0.0, 1.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Manual Encoding
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

if property_area == "Urban":
    property_area = 2
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 0

# Prediction Button
if st.button("Predict Loan Status"):

    input_data = [[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area
    ]]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")