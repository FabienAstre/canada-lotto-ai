import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see predictions.")

# Helper functions
def to_py_ticket(ticket):
    return tuple(sorted(int(x) for x in ticket))

def extract_numbers_and_bonus(df):
    required_main_cols = [
        "NUMBER DRAWN 1",
        "NUMBER DRAWN 2",
        "NUMBER DRAWN 3",
        "NUMBER DRAWN 4",
        "NUMBER DRAWN 5",
        "NUMBER DRAWN 6",
    ]
    bonus_col = "BONUS NUMBER"

    # Ensure all required columns exist
    for col in required_main_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}. Please ensure the file has 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6'.")
            return None, None, None

    # Check for invalid values and clean them
    for col in required_main_cols:
        if not df[col].between(1, 49).all():
            st.warning(f"Some values in '{col}' are out of the valid range (1-49). Cleaning data...")
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert invalid values to NaN
            df = df.dropna(subset=[col])  # Drop rows with NaN in the number columns

    # Clean the bonus column if it exists
    bonus_series = None
    if bonus_col in df.columns:
        if not df[bonus_col].between(1, 49).all():
            st.warning(f"Some values in '{bonus_col}' are out of the valid range (1-49). Cleaning data...")
            df[bonus_col] = pd.to_numeric(df[bonus_col], errors='coerce')  # Convert invalid bonus values to NaN
            df = df.dropna(subset=[bonus_col])  # Drop rows with NaN in the bonus column
        bonus_series = df[bonus_col]

    # Ensure that all the drawn numbers are within the valid range (1-49)
    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()

    # Handle dates (optional)
    date_col = None
    for col_candidate in ['DATE', 'Draw Date', 'Draw_Date', 'Date']:
        if col_candidate in df.columns:
            date_col = col_candidate
            break

    dates = None
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce')
        except:
            dates = None

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates


@st.cache_data
def compute_frequencies(numbers_df):
    all_numbers = numbers_df.values.flatten()
    return Counter(all_numbers)

# Other functions omitted for brevity...

uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to NUMBER DRAWN 6 and BONUS NUMBER",
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data (Last 30 draws):")
        st.dataframe(df.tail(30))

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)

        # If extraction fails, stop further processing
        if numbers_df is None:
            st.stop()

        # Continue with analysis...
        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]

        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))
        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter[x] if x in counter else 0)
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", title="Number Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Number Pair Frequency (last 500 draws)")
        pair_counts = compute_pair_frequencies(numbers_df, limit=500)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Frequency"])
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"({x[0]}, {x[1]})")
        fig = px.bar(pairs_df, x="Pair", y="Frequency", color="Frequency", title="Number Pair Frequency", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

        # Predictive Model
        st.subheader("Predictive Model")
        df_feat = build_prediction_features(numbers_df)
        model, accuracy = train_predictive_model(df_feat)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        probs = predict_next_draw_probs(model, numbers_df)
        st.write("Prediction probabilities for each number in the next draw:")
        st.write(probs)

        st.subheader("Ticket Generation Strategies")

        # Ticket Generation
        n_tickets = st.slider("Number of tickets to generate", 1, 100, 5)
        ticket_type = st.radio("Ticket Strategy", ["Hot/Cold Mix", "Weighted Frequency", "Smart Generation"])

        if ticket_type == "Hot/Cold Mix":
            tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
        elif ticket_type == "Weighted Frequency":
            tickets = generate_tickets_weighted(counter, n_tickets)
        else:
            fixed_nums = st.text_input("Enter fixed numbers (comma-separated, e.g., 1,2,3)")
            exclude_nums = st.text_input("Exclude recent numbers (comma-separated, e.g., 4,5,6)")
            due_nums = st.text_input("Target overdue numbers (comma-separated, e.g., 7,8,9)")
            
            fixed_nums = set(map(int, fixed_nums.split(','))) if fixed_nums else set()
            exclude_nums = set(map(int, exclude_nums.split(','))) if exclude_nums else set()
            due_nums = set(map(int, due_nums.split(','))) if due_nums else set()
            
            tickets = generate_smart_tickets(n_tickets, fixed_nums, exclude_nums, due_nums)

        st.subheader("Generated Tickets")
        st.write(tickets)

        if st.button("Download Tickets as CSV"):
            ticket_df = pd.DataFrame(tickets, columns=[f"Number {i+1}" for i in range(6)])
            csv = ticket_df.to_csv(index=False)
            st.download_button("Download", csv, "lotto_tickets.csv")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
