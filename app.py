import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ------------------------------
# Load Model & Scaler
# ------------------------------
model = load("churn_model.pkl")
scaler = load("scaler.pkl")

importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# ------------------------------
# Title
# ------------------------------

st.title("🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn")

st.markdown("""
This application predicts whether a customer is likely to churn using a trained
machine learning model and calculates the churn probability.
""")

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.header("Customer Information")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

age = st.sidebar.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    value=50000.0
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    value=50000.0
)

num_products = st.sidebar.selectbox(
    "Number of Products",
    [1,2,3,4]
)

has_card = st.sidebar.selectbox(
    "Has Credit Card",
    ["Yes","No"]
)

active_member = st.sidebar.selectbox(
    "Active Member",
    ["Yes","No"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

st.sidebar.info("Adjust customer information and click Predict Churn.")

# ------------------------------
# Encoding
# ------------------------------

has_card = 1 if has_card == "Yes" else 0

active_member = 1 if active_member == "Yes" else 0

gender_male = 1 if gender == "Male" else 0

geography_germany = 1 if geography == "Germany" else 0

geography_spain = 1 if geography == "Spain" else 0

# ------------------------------
# Feature Engineering
# ------------------------------

balance_salary_ratio = balance / (salary + 1)

product_density = num_products / (tenure + 1)

engagement = active_member * num_products

age_tenure = age * tenure

input_data = pd.DataFrame({

    "CreditScore":[credit_score],

    "Age":[age],

    "Tenure":[tenure],

    "Balance":[balance],

    "NumOfProducts":[num_products],

    "HasCrCard":[has_card],

    "IsActiveMember":[active_member],

    "EstimatedSalary":[salary],

    "BalanceSalaryRatio":[balance_salary_ratio],

    "ProductDensity":[product_density],

    "Engagement":[engagement],

    "AgeTenure":[age_tenure],

    "Geography_Germany":[geography_germany],

    "Geography_Spain":[geography_spain],

    "Gender_Male":[gender_male]

})

numerical_cols = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "Tenure"
]

# Scale only numerical columns
scaled_values = scaler.transform(input_data[numerical_cols])

input_data[numerical_cols] = pd.DataFrame(
    scaled_values,
    columns=numerical_cols,
    index=input_data.index
)


#Display the input summary

st.subheader("Customer Details")

display_data = pd.DataFrame({

    "Credit Score":[credit_score],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "Products":[num_products],
    "Salary":[salary],
    "Gender":[gender],
    "Country":[geography]

})

st.dataframe(display_data)

#Improve the prediction output

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")

    st.metric("Churn Probability", f"{probability*100:.2f}%")

    # Risk Score
    st.subheader("Risk Score")

    risk_score = probability * 100
    fig = go.Figure(go.Indicator(

    mode="gauge+number",

    value=risk_score,

    title={'text': "Customer Risk Score"},

    gauge={

        'axis': {'range': [0,100]},

        'bar': {'color': "darkred"},

        'steps':[

            {'range':[0,30],'color':"lightgreen"},

            {'range':[30,70],'color':"khaki"},

            {'range':[70,100],'color':"salmon"}

        ]

    }

))

    st.plotly_chart(fig, use_container_width=True)
    st.metric("Customer Risk Score", f"{risk_score:.2f}/100")

    st.progress(min(int(risk_score), 100))

    # Recommendation
    st.subheader("Recommended Action")

    if probability >= 0.7:
        st.error("""
High Risk Customer

Recommended Actions:
- Offer special discounts
- Relationship manager follow-up
- Personalized retention campaign
""")

    elif probability >= 0.3:
        st.warning("""
Medium Risk Customer

Recommended Actions:
- Send promotional offers
- Increase engagement
- Monitor account activity
""")

    else:
        st.success("""
Low Risk Customer

Recommended Actions:
- Continue normal customer service
- Cross-sell financial products
""")
    st.subheader("Feature Importance")

    st.bar_chart(

        importance.set_index("Feature")

    )

    report = pd.DataFrame({

        "Prediction":[

            "Churn" if prediction==1 else "No Churn"

        ],

        "Probability":[

            probability

        ]

    })

    csv = report.to_csv(index=False)

    st.download_button(

        "Download Prediction Report",

        csv,

        "prediction_report.csv",

        "text/csv"

    )

#About section

st.markdown("---")
st.subheader("About")

st.write("""
This application predicts whether a bank customer is likely to churn using a Random Forest Machine Learning model.

Features:
- Customer demographic details
- Account information
- Engineered features
- Churn probability prediction
- Risk scoring
""")

  
#Model Information

st.markdown("---")

st.subheader("Model Information")

st.write("""
**Algorithm:** Random Forest Classifier

**Dataset Size:** 10,000 Customers

**Features Used:** 15

**Target Variable:** Exited (Customer Churn)

**Feature Engineering:**
- BalanceSalaryRatio
- ProductDensity
- Engagement
- AgeTenure
""")

st.markdown("---")

st.caption(

    "Developed by Abhay Singh | MBA Business Analytics"

)