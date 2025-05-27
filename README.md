# StuAca

# Student Habits vs Academic Performance 

This project explores how lifestyle habits (like study time, sleep, diet, and social media usage) influence students' academic performance using regression and explainable AI (SHAP), with 1,000 synthetic student records.


Note: First upload from python to ipynb for github visualization, definitely could have done a better job with the plots / inputting comments below visuals rather into the code.
Update packages accordingly.

https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance

Perhaps more exploration was needed for the ML model - RF provided decent but should be compared to others such as 

- Linear Regression (Features are not highly correlated or high-dimensional here)
- Ridge (L2 Regularization) - May help here with minimizing outliers (alpha 1.0)
- Lasso (L1 Regularization) - Make some values absolutes to enable enhanced feature selection (alpha 0.1)

## Sources

This dataset is synthetic, created using Python libraries (numpy, pandas) with random distributions and logical dependencies to mimic real-life scenarios.

## Features

- Exploratory data analysis
- Linear and Random Forest regression
- SHAP explainability
- Residual and prediction diagnostics

## Files

- `stuaca.py`: python script
- `requirements.txt`: required packages
- `stuaca_notebook.ipynb`: jupyter notebook file (ignore warnings)

- Clone this and install dependencies:
```bash
pip install -r requirements.txt
python stuaca.py
