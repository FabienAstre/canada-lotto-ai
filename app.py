# --- All your original imports ---
import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
from itertools import combinations
import numpy as np

# --- New import for ML ---
from sklearn.linear_model import LogisticRegression

# --- Streamlit Config ---
st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see predictions.")

# --- Original Helper Functions ---
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
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()

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

# --- NEW: ML Model Trainer ---
@st.cache_data
def train_prediction_model(numbers_df):
    X, y = [], []
    for i in range(6, len(numbers_df)):
        prev_draws = numbers_df.iloc[i-6:i]
        current_draw = set(numbers_df.iloc[i].values)
        freq = Counter(prev_draws.values.flatten())
        features = [freq.get(n, 0) for n in range(1, 50)]
        labels = [(1 if n in current_draw else 0) for n in range(1, 50)]
        X.append(features)
        y.append(labels)

    X = np.array(X)
    y = np.array(y)

    models = []
    for n in range(49):
        model = LogisticRegression()
        model.fit(X, y[:, n])
        models.append(model)
    return models

# --- Streamlit App ---

uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to 6 and BONUS NUMBER"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(by=date_col)

        st.subheader("Uploaded Data (Last 30 draws, top = oldest):")
        st.dataframe(df.tail(30))

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("CSV must have valid columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]
        gaps = compute_number_gaps(numbers_df, dates)

        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))

        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter.get(x, 0))
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                     title="Number Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Number Pair Frequency (last 500 draws)")
        pair_counts = compute_pair_frequencies(numbers_df)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"])\
            .sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count",
                           color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        st.subheader("Number Gap Analysis")
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())})\
            .sort_values(by="Gap", ascending=False)
        overdue_threshold = st.slider("Gap threshold for overdue numbers (draws)", min_value=0, max_value=100, value=27)
        st.dataframe(gaps_df[gaps_df["Gap"] >= overdue_threshold])

        st.subheader("🎟️ Generate Lotto Tickets")
        ticket_strategy = st.selectbox(
            "Strategy for ticket generation",
            ["Pure Random", "Bias: Hot", "Bias: Cold", "Bias: Overdue", "Mixed"]
        )
        num_tickets = st.slider("How many tickets do you want to generate?", 1, 10, 5)

        def generate_ticket(pool):
            return sorted(random.sample(pool, 6))

        generated_tickets = []
        for _ in range(num_tickets):
            if ticket_strategy == "Pure Random":
                pool = list(range(1, 50))
            elif ticket_strategy == "Bias: Hot":
                pool = hot + random.sample([n for n in range(1, 50) if n not in hot], 43)
            elif ticket_strategy == "Bias: Cold":
                pool = cold + random.sample([n for n in range(1, 50) if n not in cold], 43)
            elif ticket_strategy == "Bias: Overdue":
                sorted_by_gap = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
                top_overdue = [n for n, _ in sorted_by_gap[:10]]
                pool = top_overdue + random.sample([n for n in range(1, 50) if n not in top_overdue], 39)
            elif ticket_strategy == "Mixed":
                pool = hot[:3] + cold[:2] + [n for n in range(1, 50) if n not in hot and n not in cold]
                pool += random.sample([n for n in range(1, 50) if n not in pool], max(0, 49 - len(pool)))
            else:
                pool = list(range(1, 50))

            ticket = generate_ticket(pool)
            generated_tickets.append(ticket)

        st.write("🎰 Your Generated Tickets:")
        for idx, ticket in enumerate(generated_tickets, 1):
            st.write(f"Ticket {idx}: {ticket}")

        # --- ML Prediction Section ---
        st.subheader("🧠 ML-Based Prediction (Experimental)")
        if st.button("Train Model & Predict Next Likely Numbers"):
            models = train_prediction_model(numbers_df)
            recent_draws = numbers_df.tail(6)
            freq = Counter(recent_draws.values.flatten())
            input_row = np.array([[freq.get(n, 0) for n in range(1, 50)]])
            probs = [model.predict_proba(input_row)[0][1] for model in models]
            top_predicted = np.argsort(probs)[-6:][::-1] + 1
            st.success("Predicted Most Likely Numbers:")
            st.write(sorted(top_predicted))

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
