import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
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
        order = dates.argsort()
        numbers_df = numbers_df.iloc[order].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for n in row.values:
            last_seen[n] = idx

    total_draws = len(numbers_df)
    for num in range(1, 50):
        gaps[num] = total_draws - 1 - last_seen[num] if last_seen[num] != -1 else total_draws
    return gaps

def generate_tickets_based_on_model(probs, n_tickets):
    numbers = list(range(1, 50))  # All numbers 1 to 49
    weights = np.array([probs.get(num, 0) for num in numbers])  # Get the predicted probabilities for each number
    weights = weights / weights.sum()  # Normalize the weights to make sure they sum to 1
    
    tickets = set()
    while len(tickets) < n_tickets:
        ticket = np.random.choice(numbers, 6, replace=False, p=weights)  # Randomly select 6 numbers
        tickets.add(tuple(sorted(ticket)))  # Sort and store the ticket
        
    return list(tickets)

@st.cache_data
def build_prediction_features(numbers_df, max_draws=500):
    numbers_df = numbers_df.tail(max_draws).reset_index(drop=True)
    total_draws = len(numbers_df)
    df_feat = []
    last_seen_draw = {num: -1 for num in range(1, 50)}
    freq_counter = Counter()

    for idx, row in numbers_df.iterrows():
        current_numbers = set(row.values)
        for num in range(1, 50):
            gap = idx - last_seen_draw[num] if last_seen_draw[num] != -1 else total_draws
            freq = freq_counter[num]
            df_feat.append({
                "draw_index": idx,
                "number": num,
                "gap": gap,
                "frequency": freq,
                "appeared_next": None
            })
        for num in current_numbers:
            last_seen_draw[num] = idx
            freq_counter[num] += 1

    df_feat = pd.DataFrame(df_feat)

    df_feat['appeared_next'] = 0
    for idx in range(total_draws-1):
        appeared_next_vals = set(numbers_df.iloc[idx+1].values)
        mask = df_feat['draw_index'] == idx
        df_feat.loc[mask, 'appeared_next'] = df_feat.loc[mask, 'number'].apply(lambda x: 1 if x in appeared_next_vals else 0)

    return df_feat

@st.cache_data
def train_predictive_model(df_feat):
    feature_cols = ['gap', 'frequency']
    X = df_feat[feature_cols]
    y = df_feat['appeared_next']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

def predict_next_draw_probs(model, numbers_df):
    total_draws = len(numbers_df)
    last_seen_draw = {num: -1 for num in range(1, 50)}
    freq_counter = Counter()

    for idx, row in numbers_df.iterrows():
        current_numbers = set(row.values)
        for num in current_numbers:
            last_seen_draw[num] = idx
            freq_counter[num] += 1

    features = []
    for num in range(1, 50):
        gap = total_draws - 1 - last_seen_draw[num] if last_seen_draw[num] != -1 else total_draws
        freq = freq_counter[num]
        features.append([gap, freq])

    X_pred = pd.DataFrame(features, columns=['gap', 'frequency'])
    probs = model.predict_proba(X_pred)[:,1]
    return dict(zip(range(1, 50), probs))

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
        if numbers_df is None:
            st.error("CSV must have columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        st.subheader("Predictive Model: Next Draw Number Likelihood")
        with st.spinner("Training predictive model..."):
            df_feat = build_prediction_features(numbers_df, max_draws=500)
            model, acc = train_predictive_model(df_feat)

        st.write(f"Model trained. Accuracy on test set: **{acc:.2%}**")

        # Get the probabilities of each number from the model
        probs = predict_next_draw_probs(model, numbers_df)
        
        # Show predicted probabilities
        probs_df = pd.DataFrame(list(probs.items()), columns=["Number", "Probability"]).sort_values(by="Probability", ascending=False)
        fig_pred = px.bar(probs_df, x="Number", y="Probability", title="Predicted Probability of Number in Next Draw", color="Probability", color_continuous_scale="Viridis")
        st.plotly_chart(fig_pred, use_container_width=True)

        # Ticket generation section
        budget = st.slider("Budget ($)", min_value=3, max_value=300, value=30, step=3)
        price_per_ticket = 3
        n_tickets = budget // price_per_ticket

        st.write(f"You can generate {n_tickets} tickets with your budget of ${budget}.")

        # Generate tickets based on model probabilities
        tickets = generate_tickets_based_on_model(probs, n_tickets)

        st.subheader("Generated Tickets:")
        for i, t in enumerate(tickets, 1):
            st.write(f"{i}: {t}")

        # Download button for the tickets CSV
        csv_buffer = io.StringIO()
        pd.DataFrame(tickets, columns=[f"Num {i+1}" for i in range(6)]).to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Tickets (CSV)",
            data=csv_buffer.getvalue(),
            file_name="generated_tickets.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
