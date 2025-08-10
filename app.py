import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and clean data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    # Ensure numeric columns for lotto numbers
    number_cols = [col for col in df.columns if col.lower().startswith('num')]
    for col in number_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=number_cols, inplace=True)
    return df, number_cols

# Machine Learning prediction function
def predict_numbers(df, number_cols):
    # Prepare training data
    X = []
    y = []
    for col in number_cols:
        col_nums = df[col].values
        for num in range(1, 50):
            X.append([num])
            y.append(1 if num in col_nums else 0)
    
    X = np.array(X)
    y = np.array(y)

    model = LogisticRegression()
    model.fit(X, y)
    
    # Predict probabilities
    probs = model.predict_proba(np.arange(1, 50).reshape(-1, 1))[:, 1]
    
    # Pick top 6 numbers
    top_numbers = np.argsort(probs)[-6:][::-1] + 1
    return list(map(int, top_numbers))  # convert np.int64 to int

# Generate ML tickets
def generate_ml_tickets(base_numbers, n_tickets=3):
    tickets = []
    for _ in range(n_tickets):
        ticket = base_numbers.copy()
        while len(ticket) < 6:
            rand_num = np.random.randint(1, 50)
            if rand_num not in ticket:
                ticket.append(rand_num)
        np.random.shuffle(ticket)
        tickets.append(list(map(int, ticket)))  # convert to plain int
    return tickets

# Streamlit app
st.title("Lotto 6/49 Analyzer with ML Predictions")

uploaded_file = st.file_uploader("Upload your Lotto CSV file", type=["csv"])

if uploaded_file is not None:
    df, number_cols = load_data(uploaded_file)

    # Show data
    with st.expander("View Uploaded Data"):
        st.dataframe(df)

    # ML prediction
    base_predicted_numbers = predict_numbers(df, number_cols)
    
    st.subheader("Base Predicted Numbers (most common 6):")
    st.write(base_predicted_numbers)

    # Generated ML tickets
    ml_tickets = generate_ml_tickets(base_predicted_numbers, n_tickets=3)
    
    st.subheader("Generated ML Tickets:")
    for i, ticket in enumerate(ml_tickets, start=1):
        st.write(f"ML Ticket {i}: {ticket}")
