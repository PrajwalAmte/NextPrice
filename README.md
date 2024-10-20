# 🌍 NextPrice

**NextPrice** is a powerful web application designed to help individuals and businesses predict future living costs and inflation trends. The tool provides two key features: projecting the cost of living in a single country based on inflation, and comparing relative inflation trends between two countries over multiple years.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contributors](#contributors)

---

## ✨ Features

### Tab 1: **Cost of Living Predictor**
- 📈 Predict the cost of living in a specific country for a future year.
- 📊 Uses compounded inflation to give an accurate forecast of future living expenses.
- 💼 Suitable for individuals planning financial goals, retirement, or future budgets.

### Tab 2: **Relative Inflation Comparison**
- 🌍 Compare inflation trends between two countries over multiple years.
- 🔎 Understand economic differences for international planning or investment decisions.
- 💡 Highlights relative inflation trends, providing insights into future purchasing power in different regions.

---

## ⚙️ Installation

Follow these steps to get **NextPrice** up and running locally:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/nextprice.git
    cd nextprice
    ```

2. **Install the required dependencies**:

    Make sure you have **Python 3.6+** installed. Then, install the dependencies listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:

    Start the Streamlit app using the following command:

    ```bash
    streamlit run app.py
    ```

4. **Access the web app**:

    Open your browser and navigate to `http://localhost:8501/` to use **NextPrice**.

---

## 🛠️ How to Use

### Tab 1: **Predicting Future Cost of Living**

1. Select your country.
2. Enter the current year and the target year for the prediction.
3. Input your current cost of living.
4. The app will use compounded inflation to forecast your future cost of living.

### Tab 2: **Comparing Relative Inflation**

1. Choose two countries for comparison.
2. Input the current year and target year.
3. The tool will predict inflation for both countries and display their inflation trends across multiple years.
4. The relative inflation difference between the two countries will also be shown for an easy comparison of future purchasing power.

---

## 📂 Project Structure

```bash
nextprice/
│
├── app.py                     # Main application code for Streamlit
├── global_inflation_data.csv   # Dataset used for inflation prediction
├── requirements.txt            # List of required Python packages
├── README.md                   # Project documentation
└── utils.py                    # Helper functions for data processing and inflation prediction
```

---

## 🧰 Technologies Used

- **Python 3.6+** - Main programming language
- **Streamlit** - Web framework for building the interactive application
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning for regression models
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

---

## 📸 Screenshots

### Cost of Living Predictor (Tab 1):

*Include a screenshot of the cost of living prediction tab here.*

### Relative Inflation Comparison (Tab 2):

*Include a screenshot of the relative inflation comparison tab here.*

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
