import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- Streamlit Page Config ---
st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see ML predictions.")

# --- Helper Functions ---
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
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df):
    return Counter(numbers_df.values.flatten())

@st.cache_data
def compute_pair_frequencies(numbers_df, limit=500):
    pair_counts = Counter()
    df_subset = numbers_df.tail(limit)
    for _, row in df_subset.iterrows():
        pairs = combinations(sorted(row.values), 2)
        pair_counts.update(pairs)
    return pair_counts

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {n: -1 for n in range(1, 50)}
    gaps = {}

    if dates is not None:
        numbers_df = numbers_df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for n in row.values:
            last_seen[n] = idx

    total_draws = len(numbers_df)
    for n in range(1, 50):
        gaps[n] = total_draws - 1 - last_seen[n] if last_seen[n] != -1 else total_draws
    return gaps

@st.cache_data
def train_prediction_model(numbers_df):
    X, y = [], []

    for i in range(6, len(numbers_df)):
        prev_draws = numbers_df.iloc[i - 6:i]
        current_draw = set(numbers_df.iloc[i].values)

        freq = Counter(prev_draws.values.flatten())
        row = [freq.get(n, 0) for n in range(1, 50)]
        label_row = [1 if n in current_draw else 0 for n in range(1, 50)]

        X.append(row)
        y.append(label_row)

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(49):
        model = LogisticRegression()
        model.fit(X, y[:, i])
        models.append(model)

    return models

# --- App UI ---
uploaded_file = st.file_uploader("📂 Upload Lotto 6/49 CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.sort_values(by=date_col).reset_index(drop=True)

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("❌ Your CSV must contain valid 'NUMBER DRAWN 1–6' columns with values between 1 and 49.")
            st.stop()

        # ✅ Display last 30 draws including the DATE
        st.subheader("📅 Uploaded Data (Last 30 draws, top = oldest):")
        preview_df = df.tail(30)
        st.dataframe(preview_df)

        # 🔥 Hot & Cold Numbers
        freq_counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in freq_counter.most_common(6)]
        cold = [num for num, _ in freq_counter.most_common()[:-7:-1]]
        gaps = compute_number_gaps(numbers_df, dates)

        st.subheader("🔥 Hot Numbers")
        st.write(", ".join(map(str, hot)))

        st.subheader("❄️ Cold Numbers")
        st.write(", ".join(map(str, cold)))

        # 📊 Frequency Chart
        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: freq_counter.get(x, 0))
        fig_freq = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", title="Number Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig_freq, use_container_width=True)

        # 📊 Pair Frequency
        st.subheader("👥 Top Number Pairs (last 500 draws)")
        pair_counts = compute_pair_frequencies(numbers_df)
        pair_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"]).sort_values("Count", ascending=False).head(20)
        pair_df["Pair"] = pair_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pair_df, y="Pair", x="Count", orientation='h', color="Count", color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        # 📉 Gap Analysis
        st.subheader("⏳ Number Gap Analysis")
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values(by="Gap", ascending=False)
        gap_threshold = st.slider("Highlight overdue numbers with gap ≥", 0, 100, 27)
        st.dataframe(gaps_df[gaps_df["Gap"] >= gap_threshold])

        # 🎟️ Ticket Generator
        st.subheader("🎟️ Generate Lotto Tickets")
        strategy = st.selectbox("Strategy", ["Pure Random", "Bias: Hot", "Bias: Cold", "Bias: Overdue", "Mixed"])
        num_tickets = st.slider("How many tickets?", 1, 10, 5)

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
                overdue_sorted = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
                pool = [n for n, _ in overdue_sorted[:10]] + random.sample([n for n in range(1, 50) if n not in [x for x, _ in overdue_sorted[:10]]], 39)
            elif strategy == "Mixed":
                pool = hot[:3] + cold[:2] + [n for n in range(1, 50) if n not in hot and n not in cold]
                pool += random.sample([n for n in range(1, 50) if n not in pool], max(0, 49 - len(pool)))
            else:
                pool = list(range(1, 50))

            ticket = generate_ticket(pool)
            generated_tickets.append([int(n) for n in ticket])  # convert to Python int

        st.write("🎰 Your Generated Tickets:")
        for i, t in enumerate(generated_tickets, 1):
            st.write(f"Ticket {i}: {t}")

        # 🧠 ML Prediction
        st.subheader("🧠 ML-Based Prediction (Experimental)")
        if st.button("Train & Predict Next Likely Numbers"):
            models = train_prediction_model(numbers_df)
            recent_draws = numbers_df.tail(6)
            freq = Counter(recent_draws.values.flatten())
            input_row = np.array([[freq.get(n, 0) for n in range(1, 50)]])
            probs = [model.predict_proba(input_row)[0][1] for model in models]
            top_predicted = np.argsort(probs)[-6:][::-1] + 1
            st.success("Predicted Numbers:")
            st.write(sorted(top_predicted))

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
else:
    st.info("Please upload a valid Lotto 6/49 CSV file.")
