# House Price Prediction App

## Project Overview

The **House Price Prediction App** is a web application built using Streamlit that predicts house prices based on various features derived from the Boston Housing dataset. This application leverages machine learning algorithms to provide accurate predictions, allowing users to input housing features and receive instant feedback on predicted house prices.

## Dataset

The project utilizes the **Boston Housing dataset** sourced from Kaggle. The dataset contains information about housing values in suburbs of Boston, including various features that affect house prices.

## Algorithms Used

Three regression algorithms were implemented to predict house prices:

1. **Linear Regression**
   - MSE: 26.27
   - R²: 0.64

2. **Random Forest**
   - MSE: 10.24
   - R²: 0.86

3. **Decision Tree Regression**
   - MSE: 13.00
   - R²: 0.82

After evaluating the models, the **Random Forest** model was chosen for deployment due to its superior performance in both MSE and R² metrics.

## Features

- User-friendly interface to input housing features
- Predict house prices in US dollars
- Visualizations for feature importance and predicted vs actual prices

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/house-price-prediction-app.git
## Navigate the project directory 
cd house-price-prediction-app
## Create a virtual enivornment 
python -m venv venv
## Usage
To run the app, execute the following command in your terminal:  streamlit run app.py
## License
This project is licensed under the MIT License. See the LICENSE file for more details
## Acknowledgments
Boston Housing dataset from Kaggle
Streamlit for the web app framework
Scikit-learn for machine learning algorithms

## Screen Shot 

![Screenshot 2024-10-19 151314](https://github.com/user-attachments/assets/4f2ab9ba-1c4a-4ae7-96e1-46463d598a0f)

![Screenshot 2024-10-19 151403](https://github.com/user-attachments/assets/799e5aba-49d7-42ad-9888-3a99bd4d8f7a)


![Screenshot 2024-10-19 151805](https://github.com/user-attachments/assets/fa665afe-c460-4f7c-ad83-8c8719c1ee55)


