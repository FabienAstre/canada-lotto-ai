import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lotto 6/49 Analyzer", layout="wide")
st.title("🎰 Lotto 6/49 Analyzer + ML Predictor")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Lotto 6/49 CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Preprocessing ---
    number_cols = [col for col in df.columns if any(str(i) in col for i in range(1, 50)) or "NUMBER DRAWN" in col.upper()]
    number_cols = number_cols[:6]  # Assume first 6 are draw numbers

    numbers_df = df[number_cols].dropna().astype(int)

    # --- Frequency Analysis ---
    all_numbers = numbers_df.values.flatten()
    freq_counter = Counter(all_numbers)
    freq_series = pd.Series(freq_counter).sort_index()

    # --- Gap Analysis ---
    def compute_number_gaps(df):
        last_seen = {i: None for i in range(1, 50)}
        gaps = {i: None for i in range(1, 50)}
        for index in reversed(df.index):
            row = df.loc[index].values
            for n in range(1, 50):
                if last_seen[n] is None and n in row:
                    last_seen[n] = index
        for n in range(1, 50):
            if last_seen[n] is not None:
                gaps[n] = df.index[-1] - last_seen[n]
            else:
                gaps[n] = None
        return gaps

    gaps = compute_number_gaps(numbers_df)

    # --- Pair Frequency ---
    pair_counter = Counter()
    for row in numbers_df.values:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                pair = tuple(sorted((row[i], row[j])))
                pair_counter[pair] += 1
    top_pairs = pair_counter.most_common(10)

    # --- Layout Columns ---
    col1, col2 = st.columns(2)

    # --- Frequency Chart ---
    with col1:
        st.subheader("Number Frequency")
        fig = go.Figure(data=[go.Bar(x=freq_series.index, y=freq_series.values)])
        st.plotly_chart(fig, use_container_width=True)

    # --- Gap Chart ---
    with col2:
        st.subheader("Number Gaps (Draws since last appearance)")
        gap_series = pd.Series(gaps).sort_index()
        fig2 = go.Figure(data=[go.Bar(x=gap_series.index, y=gap_series.values)])
        st.plotly_chart(fig2, use_container_width=True)

    # --- Pair Frequencies ---
    st.subheader("Top 10 Frequent Pairs")
    for pair, count in top_pairs:
        st.write(f"{pair}: {count} times")

    # --- Data Table (Last 30 Draws) ---
    st.subheader("Uploaded Data (Last 30 draws, top = oldest):")
    date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(by=date_col)
        display_cols = [date_col] + number_cols
        st.dataframe(df[display_cols].tail(30).reset_index(drop=True))
    else:
        st.dataframe(df[number_cols].tail(30).reset_index(drop=True))

    # --- Ticket Generator ---
    st.subheader("🎟️ Generate Random Tickets")
    ticket_mode = st.selectbox("Select ticket mode:", ["Random", "Frequency Biased"])
    num_tickets = st.slider("How many tickets to generate?", 1, 10, 3)

    def generate_ticket(biased=False):
        if biased:
            weights = np.array([freq_counter.get(n, 0.1) for n in range(1, 50)])
            ticket = np.random.choice(range(1, 50), size=6, replace=False, p=weights/weights.sum())
        else:
            ticket = np.random.choice(range(1, 50), size=6, replace=False)
        return sorted([int(n) for n in ticket])

    generated_tickets = [generate_ticket(biased=(ticket_mode == "Frequency Biased")) for _ in range(num_tickets)]
    for idx, ticket in enumerate(generated_tickets, 1):
        st.write(f"Ticket {idx}: {ticket}")

    # --- ML Prediction ---
    st.subheader("🧠 ML-Based Prediction (Experimental)")

    def train_prediction_model(numbers_df):
        X = []
        y = []

        for i in range(6, len(numbers_df)):
            prev_draws = numbers_df.iloc[i-6:i]
            current_draw = set(numbers_df.iloc[i].values)

            freq = Counter(prev_draws.values.flatten())
            row = [freq.get(n, 0) for n in range(1, 50)]
            X.append(row)

            y_row = [(1 if n in current_draw else 0) for n in range(1, 50)]
            y.append(y_row)

        X = np.array(X)
        y = np.array(y)

        models = []
        for n in range(49):
            model = LogisticRegression()
            model.fit(X, y[:, n])
            models.append(model)

        return models

    if st.button("Train Model & Predict Next Likely Numbers"):
        models = train_prediction_model(numbers_df)
        recent_draws = numbers_df.tail(6)
        freq = Counter(recent_draws.values.flatten())
        input_row = np.array([[freq.get(n, 0) for n in range(1, 50)]])

        probs = [model.predict_proba(input_row)[0][1] for model in models]
        top_predicted = np.argsort(probs)[-6:][::-1] + 1

        st.write("Predicted Numbers:", sorted([int(n) for n in top_predicted]))
