import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
import plotly.express as px

# Streamlit config
st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see predictions.")

# --- Helper Functions ---

def to_py_ticket(ticket):
    return tuple(sorted(int(x) for x in ticket))

def extract_numbers_and_bonus(df):
    required_main_cols = [
        "NUMBER DRAWN 1", "NUMBER DRAWN 2", "NUMBER DRAWN 3",
        "NUMBER DRAWN 4", "NUMBER DRAWN 5", "NUMBER DRAWN 6",
    ]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in required_main_cols):
        return None, None, None

    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)
    dates = None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        dates = df[date_col]

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df):
    all_numbers = numbers_df.values.flatten()
    return Counter(all_numbers)

@st.cache_data
def compute_pair_frequencies(numbers_df, limit=500):
    pair_counts = Counter()
    df_subset = numbers_df.tail(limit)
    for _, row in df_subset.iterrows():
        pairs = combinations(sorted(row.values), 2)
        pair_counts.update(pairs)
    return pair_counts

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {num: -1 for num in range(1, 50)}
    gaps = {num: None for num in range(1, 50)}

    if dates is not None:
        numbers_df = numbers_df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for n in row.values:
            last_seen[n] = idx

    total_draws = len(numbers_df)
    for num in range(1, 50):
        gaps[num] = total_draws - 1 - last_seen[num] if last_seen[num] != -1 else total_draws
    return gaps

# --- App Main ---

uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to 6, BONUS NUMBER, and optional DATE"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # --- ADJUSTED UPLOADED DATA DISPLAY ---
        # Find date column
        date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(by=date_col)

        # Prepare columns to display including date column explicitly
        columns_to_display = df.columns.tolist()
        if date_col and date_col not in columns_to_display:
            columns_to_display.insert(0, date_col)  # put date first

        st.subheader("Uploaded Data (Last 30 draws, top = oldest):")
        st.dataframe(df.tail(30)[columns_to_display].reset_index(drop=True))
        # --- END ADJUSTED DISPLAY ---

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("CSV must have valid columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        # Frequencies
        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]
        gaps = compute_number_gaps(numbers_df, dates)

        # Hot & Cold
        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))

        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        # Frequency Chart
        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter.get(x, 0))
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                     title="Number Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        # Pair Frequency
        st.subheader("Number Pair Frequency (last 500 draws)")
        pair_counts = compute_pair_frequencies(numbers_df)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"])\
            .sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count",
                           color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        # Gap Analysis
        st.subheader("Number Gap Analysis")
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())})\
            .sort_values(by="Gap", ascending=False)
        threshold = st.slider("Gap threshold for overdue numbers (draws)", min_value=0, max_value=100, value=27)
        st.dataframe(gaps_df[gaps_df["Gap"] >= threshold])

        # Ticket Generator
        st.subheader("🎟️ Generate Lotto Tickets")

        strategy = st.selectbox(
            "Strategy for ticket generation",
            ["Pure Random", "Bias: Hot", "Bias: Cold", "Bias: Overdue", "Mixed"]
        )
        num_tickets = st.slider("How many tickets do you want to generate?", 1, 10, 5)

        def generate_ticket(pool):
            return sorted(random.sample(pool, 6))

        generated_tickets = []
        for _ in range(num_tickets):
            if strategy == "Pure Random":
                pool = list(range(1, 50))
            elif strategy == "Bias: Hot":
                pool = hot + random.sample([n for n in range(1, 50) if n not in hot], 43)
            elif strategy == "Bias: Cold":
                pool = cold + random.sample([n for n in range(1, 50) if n not in cold], 43)
            elif strategy == "Bias: Overdue":
                sorted_by_gap = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
                top_gap = [n for n, g in sorted_by_gap[:10]]
                pool = top_gap + random.sample([n for n in range(1, 50) if n not in top_gap], 39)
            elif strategy == "Mixed":
                pool = hot[:3] + cold[:2]
                pool += random.sample([n for n in range(1, 50) if n not in pool], 49 - len(pool))
            else:
                pool = list(range(1, 50))
            ticket = generate_ticket(pool)
            ticket_clean = [int(n) for n in ticket]
            generated_tickets.append(ticket_clean)

        st.write("🎰 Your Generated Tickets:")
        for idx, ticket in enumerate(generated_tickets, 1):
            st.write(f"Ticket {idx}: {ticket}")

        # 🧠 ML-based Prediction
        st.subheader("🧠 ML-Based Prediction (Experimental)")

        most_common = counter.most_common(6)
        predicted_numbers = sorted([int(num) for num, _ in most_common])

        st.write("Predicted Numbers:")
        st.write(predicted_numbers)

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
