import pandas as pd 
import numpy as np 
df = pd.read_csv('data/diabetes.csv')

zero_missing_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for col in zero_missing_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

print("Missing Values Handled")

print(df.isna().sum())    
    
