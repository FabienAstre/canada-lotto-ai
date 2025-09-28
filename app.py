import pandas as pd

# Load CSV
df = pd.read_csv("lottery.csv")

# Remove non-numeric characters from number columns
number_cols = [f'NUMBER DRAWN {i}' for i in range(1, 7)] + ['Bonus']
for col in number_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')

# Check for missing values
print(df[number_cols].isna().sum())

# Fill missing numbers with 0 or drop rows with missing values
df[number_cols] = df[number_cols].fillna(0).astype(int)
