import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

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


# Function to train a linear regression model and predict inflation
def predict_inflation(df, country, year_range):
    inflation_data = prepare_country_data(df, country)

    # Train a linear regression model
    X = inflation_data[['Year']]  # Feature: Year
    y = inflation_data['Inflation']  # Target: Inflation rate

    model = LinearRegression()
    model.fit(X, y)

    # Predict inflation for the range of years
    future_years = np.array(year_range).reshape(-1, 1)
    predictions = model.predict(future_years)

    return predictions, inflation_data


# Streamlit UI with Tabs
st.title("Cost of Living Prediction Tool")

# Creating two tabs: one for the first project and another for the second project
tab1, tab2 = st.tabs(["Cost of Living Prediction", "Relative Inflation Between Countries"])

# First Tab: Predicting cost of living in the same country over different years
with tab1:
    st.header("Cost of Living Prediction")

    # User inputs
    country = st.text_input("Enter the country:")
    current_year = st.number_input("Enter the current year:", min_value=2000, max_value=2100, value=2024)
    future_year = st.number_input("Enter the target year:", min_value=2000, max_value=2100, value=2028)
    current_cost = st.number_input("Enter your current cost of living:", min_value=0.0)

    if st.button("Predict Cost of Living"):
        if country and current_cost > 0:
            # Get inflation predictions for current and future years
            years_range = list(range(current_year, future_year + 1))
            future_inflation, inflation_data = predict_inflation(df, country, years_range)

            # Calculate the number of years between current and future year
            num_years = future_year - current_year

            # Calculate the projected cost of living with compounded inflation
            projected_cost = current_cost * ((1 + future_inflation[-1] / 100) ** num_years)

            # Display results
            st.write(f"Predicted inflation in {country} for {current_year} is: {future_inflation[0]:.2f}%")
            st.write(f"Predicted inflation in {country} for {future_year} is: {future_inflation[-1]:.2f}%")
            st.write(
                f"Your projected cost of living in {country} for {future_year} is: {projected_cost:.2f} (from {current_cost:.2f})")

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
            current_country_inflation, _ = predict_inflation(df, current_country, years_range)
            future_country_inflation, _ = predict_inflation(df, future_country, years_range)

            # Calculate relative inflation for each year in the range
            relative_inflation = future_country_inflation - current_country_inflation

            # Display results
            st.write(
                f"Predicted inflation in {current_country} for {future_year} is: {current_country_inflation[-1]:.2f}%")
            st.write(
                f"Predicted inflation in {future_country} for {future_year} is: {future_country_inflation[-1]:.2f}%")
            st.write(
                f"Relative inflation (Future Country - Current Country) for {future_year} is: {relative_inflation[-1]:.2f}%")

            # Plot relative inflation trend over the years
            plt.figure(figsize=(10, 5))

            # Plot the relative inflation trend, using color to show positive and negative differences
            plt.fill_between(years_range, relative_inflation, 0, where=(relative_inflation > 0), color='green',
                             alpha=0.6, label='Positive Difference')
            plt.fill_between(years_range, relative_inflation, 0, where=(relative_inflation < 0), color='red', alpha=0.6,
                             label='Negative Difference')
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