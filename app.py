import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('global_inflation_data.csv')

# Function to prepare data for a specific country
def prepare_country_data(df, country):
    country_data = df[df['country_name'] == country]

    # Assuming the columns are years (without month information)
    inflation_data = country_data.iloc[:, 2:].T  # Exclude country_name and indicator columns
    inflation_data.columns = ['Inflation']

    # Convert the index (which contains the years) to a 'Year' column
    inflation_data['Year'] = inflation_data.index.astype(int)

    return inflation_data

# Function to train a Random Forest model and predict inflation
def predict_inflation_rf(df, country, year_range):
    inflation_data = prepare_country_data(df, country)

    # Prepare data for Random Forest
    X = inflation_data[['Year']].astype(int)  # Feature: Year
    y = inflation_data['Inflation'].astype(float)  # Target: Inflation rate

    # Split the data into train and test sets
    train_size = int(0.8 * len(X))  # Using 80% for training and 20% for testing
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict future inflation
    future_years = np.array(year_range).reshape(-1, 1)
    predictions = rf_model.predict(future_years)

    # Calculate accuracy metrics for the training set
    y_pred_train = rf_model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    accuracy_train = 100 * (1 - mse_train / np.var(y_train))

    # Calculate accuracy metrics for the test set
    y_pred_test = rf_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    accuracy_test = 100 * (1 - mse_test / np.var(y_test))

    return predictions, inflation_data, mse_train, accuracy_train, mse_test, accuracy_test

# Streamlit UI with Tabs
st.title("Cost of Living Prediction Tool")

# Creating two tabs: one for the first project and another for the second project
tab1, tab2 = st.tabs(["Same Country Prediction", "Relative Inflation Between Countries"])

# First Tab: Predicting cost of living in the same country over different years
with tab1:
    st.header("Cost of Living Prediction (Same Country)")

    # User inputs
    country = st.text_input("Enter the country:")
    current_year = st.number_input("Enter the current year:", min_value=2000, max_value=2100, value=2024)
    future_year = st.number_input("Enter the target year:", min_value=2000, max_value=2100, value=2028)
    current_cost = st.number_input("Enter your current cost of living:", min_value=0.0)

    if st.button("Predict Cost of Living"):
        if country and current_cost > 0:
            # Get inflation predictions for current and future years
            years_range = list(range(current_year, future_year + 1))
            future_inflation, inflation_data, mse_train, acc_train, mse_test, acc_test = predict_inflation_rf(df, country, years_range)

            # Display MSE and Accuracy
            st.write(f"Mean Squared Error (Train Set): {mse_train:.2f}")
            st.write(f"Accuracy Percentage (Train Set): {acc_train:.2f}%")
            st.write(f"Mean Squared Error (Test Set): {mse_test:.2f}")
            st.write(f"Accuracy Percentage (Test Set): {acc_test:.2f}%")

            # Calculate the number of years between current and future year
            num_years = future_year - current_year

            # Calculate the projected cost of living with compounded inflation
            projected_cost = current_cost * ((1 + future_inflation[-1] / 100) ** num_years)

            # Display results
            st.write(f"Predicted inflation in {country} for {current_year} is: {future_inflation[0]:.2f}%")
            st.write(f"Predicted inflation in {country} for {future_year} is: {future_inflation[-1]:.2f}%")
            st.write(f"Your projected cost of living in {country} for {future_year} is: {projected_cost:.2f} (from {current_cost:.2f})")

            # Plotting historical inflation data
            plt.figure(figsize=(10, 5))
            plt.plot(inflation_data['Year'], inflation_data['Inflation'], marker='o', label='Historical Inflation')
            plt.plot(years_range, future_inflation, marker='x', linestyle='--', label='Predicted Inflation')
            plt.axvline(current_year, color='blue', linestyle='--', label=f'Current Year: {current_year}')
            plt.axvline(future_year, color='red', linestyle='--', label=f'Target Year: {future_year}')
            plt.title(f'Inflation Rates in {country} (Historical and Predicted)')
            plt.xlabel('Year')
            plt.ylabel('Inflation Rate (%)')
            plt.legend()
            plt.grid()

            # Display the plot in Streamlit
            st.pyplot(plt)
        else:
            st.error("Please enter valid inputs for country and cost of living.")

# Second Tab: Calculating relative inflation between two different countries
with tab2:
    st.header("Relative Inflation Between Countries")

    # User inputs for relative inflation prediction
    current_country = st.text_input("Enter your current country:")
    future_country = st.text_input("Enter the target country:")
    current_year = st.number_input("Enter the current year for inflation:", min_value=2000, max_value=2100, value=2024)
    future_year = st.number_input("Enter the target year for inflation:", min_value=2000, max_value=2100, value=2028)

    if st.button("Predict Relative Inflation"):
        if current_country and future_country:
            # Get inflation predictions for both countries over the range of years
            years_range = list(range(current_year, future_year + 1))
            current_country_inflation, _, _, _, _, _ = predict_inflation_rf(df, current_country, years_range)
            future_country_inflation, _, _, _, _, _ = predict_inflation_rf(df, future_country, years_range)

            # Calculate relative inflation for each year in the range
            relative_inflation = future_country_inflation - current_country_inflation

            # Display results
            st.write(f"Predicted inflation in {current_country} for {future_year} is: {current_country_inflation[-1]:.2f}%")
            st.write(f"Predicted inflation in {future_country} for {future_year} is: {future_country_inflation[-1]:.2f}%")
            st.write(f"Relative inflation (Future Country - Current Country) for {future_year} is: {relative_inflation[-1]:.2f}%")

            # Plot relative inflation trend over the years
            plt.figure(figsize=(10, 5))

            # Plot the relative inflation trend, using color to show positive and negative differences
            plt.fill_between(years_range, relative_inflation, 0, where=(relative_inflation > 0), color='green', alpha=0.6, label='Positive Difference')
            plt.fill_between(years_range, relative_inflation, 0, where=(relative_inflation < 0), color='red', alpha=0.6, label='Negative Difference')
            plt.plot(years_range, relative_inflation, marker='o', color='purple', label='Relative Inflation Trend')

            plt.title('Relative Inflation Trend Between Countries')
            plt.xlabel('Year')
            plt.ylabel('Relative Inflation (%)')
            plt.legend()
            plt.grid()

            # Display the plot in Streamlit
            st.pyplot(plt)
        else:
            st.error("Please fill in all the fields for relative inflation prediction.")
