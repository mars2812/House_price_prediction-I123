import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

# Load the saved Random Forest model
model_rf = joblib.load('random_forest_model.pkl')

# Define a function to make predictions based on user input
def predict_house_price(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model_rf.predict(input_array)
    return prediction[0]

# Streamlit app
st.set_page_config(page_title="House Price Prediction App", page_icon="🏠")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select an option:", ["Predict House Price", "About this App"])

if options == "About this App":
    st.title("About this App")
    st.markdown("""
    This application leverages the Boston Housing dataset sourced from Kaggle, where extensive exploratory data analysis (EDA) and preprocessing have been conducted to prepare the data for modeling.

    To predict house prices, three regression algorithms were applied: Random Forest, Linear Regression, and Decision Tree Regression. The models were evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics to assess their performance.

    The evaluation results are as follows:

    - **Linear Regression**: 
      - MSE: 26.27 
      - R²: 0.64 
    - **Random Forest**: 
      - MSE: 10.24 
      - R²: 0.86 
    - **Decision Tree Regression**: 
      - MSE: 13.00 
      - R²: 0.82 

    Based on these results, the **Random Forest** model was selected for deployment due to its superior performance in both MSE and R² metrics. This application provides an intuitive interface for users to input housing features and receive accurate predictions for house prices.
    """)
else:
    # Main App Styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f0f2f5;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
        }
        .title-background {
            background-color: #007BFF; /* Change to your desired color */
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            color: white; /* Change title text color */
        }
        h1 {
            margin: 0; /* Remove default margin */
            font-family: 'Arial', sans-serif;
        }
        h2 {
            color: #4B0082;
        }
        .prediction {
            font-size: 24px;
            color: #007BFF;
            font-weight: bold;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            font-size: 14px;
            color: #666;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='title-background'><h1>House Price Prediction App</h1></div>", unsafe_allow_html=True)

    st.markdown("<div class='main'>", unsafe_allow_html=True)

    st.header("Input the following details to predict the house price:")

    # Input fields
    CRIM = st.number_input("CRIM (per capita crime rate by town)", min_value=0.0, step=0.001)
    ZN = st.number_input("ZN (proportion of residential land zoned for lots over 25,000 sq. ft.)", min_value=0.0, step=0.1)
    INDUS = st.number_input("INDUS (proportion of non-retail business acres per town)", min_value=0.0, step=0.1)
    CHAS = st.selectbox("CHAS (Charles River dummy variable)", options=[0, 1])
    NOX = st.number_input("NOX (nitric oxides concentration)", min_value=0.0, step=0.001)
    RM = st.number_input("RM (average number of rooms per dwelling)", min_value=0.0, step=0.1)
    AGE = st.number_input("AGE (proportion of owner-occupied units built prior to 1940)", min_value=0.0, step=0.1)
    DIS = st.number_input("DIS (weighted distances to five Boston employment centres)", min_value=0.0, step=0.1)
    RAD = st.number_input("RAD (index of accessibility to radial highways)", min_value=1, step=1)
    TAX = st.number_input("TAX (full-value property-tax rate per $10,000)", min_value=1, step=1)
    PTRATIO = st.number_input("PTRATIO (pupil-teacher ratio by town)", min_value=1.0, step=0.1)

    # Predict button
    if st.button("Predict House Price"):
        input_data = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO]
        
        predicted_price = predict_house_price(input_data)
        
        # Display the result in US dollars
        st.markdown(f"<div class='prediction'>The predicted house price: ${predicted_price * 1000:.2f}</div>", unsafe_allow_html=True)

        # Feature importance visualization
        feature_importances = model_rf.feature_importances_
        feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO"]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_names)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        st.pyplot(plt)

        # Interactive scatter plot of predicted vs. actual prices
        # Assume actual_prices is available from your dataset for comparison
        # For demonstration, generating dummy data
        actual_prices = np.random.uniform(10, 50, size=100)  # Replace this with actual data
        predicted_prices = np.random.uniform(10, 50, size=100)  # Replace with your predicted prices

        df = pd.DataFrame({'Actual Price': actual_prices, 'Predicted Price': predicted_prices})
        fig = px.scatter(df, x='Actual Price', y='Predicted Price', title='Predicted vs Actual House Prices', labels={'Actual Price': 'Actual Price ($1000s)', 'Predicted Price': 'Predicted Price ($1000s)'})
        st.plotly_chart(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
   # Footer
# Footer
st.markdown("<div class='footer' style='color: green;'><strong>Developed by Matrika Dhamala. &copy; 2024. All rights reserved.</strong></div>", unsafe_allow_html=True)
