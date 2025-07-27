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

def generate_tickets_hot_cold(hot, cold, n_tickets):
    tickets = set()
    while len(tickets) < n_tickets:
        n_hot = random.randint(2, min(4, len(hot)))
        n_cold = random.randint(2, min(4, len(cold)))
        pick_hot = random.sample(hot, n_hot)
        pick_cold = random.sample(cold, n_cold)
        current = set(pick_hot + pick_cold)
        while len(current) < 6:
            current.add(random.randint(1, 49))
        tickets.add(to_py_ticket(current))
    return list(tickets)

def generate_tickets_weighted(counter, n_tickets):
    numbers = np.array(range(1, 50))
    freqs = np.array([counter.get(num, 0) for num in numbers])
    weights = freqs + 1
    tickets = set()
    while len(tickets) < n_tickets:
        ticket_np = np.random.choice(numbers, 6, replace=False, p=weights/weights.sum())
        tickets.add(to_py_ticket(ticket_np))
    return list(tickets)

def generate_smart_tickets(n_tickets, fixed_nums, exclude_nums, due_nums):
    tickets = set()
    pool = set(range(1, 50)) - exclude_nums - fixed_nums
    while len(tickets) < n_tickets:
        ticket = set(fixed_nums)
        for num in (due_nums - ticket):
            if len(ticket) < 6:
                ticket.add(num)
        remaining_pool = list(pool - ticket)
        random.shuffle(remaining_pool)
        for num in remaining_pool:
            if len(ticket) >= 6:
                break
            ticket.add(num)
        tickets.add(to_py_ticket(ticket))
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
        # Read the CSV into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Reverse the DataFrame rows (from bottom to top)
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        
        st.subheader("Uploaded Data (Reversed):")
        st.dataframe(df_reversed.tail(30))  # Display the last 30 rows of the reversed data

        # Extract numbers and bonus from the reversed data
        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df_reversed)
        if numbers_df is None:
            st.error("CSV must have columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]

        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))
        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        # Continue with the rest of the code...
        # All code below will remain unchanged.
