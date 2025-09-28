import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from itertools import combinations
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# ======================
# Streamlit Page Config
# ======================
st.set_page_config(
    page_title="🎲 Canada Lotto 6/49 Analyzer",
    page_icon="🎲",
    layout="wide",
)
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, simulate strategies, and ML-based predictions.")

# ======================
# Helper Functions
# ======================

def extract_numbers_and_bonus(df: pd.DataFrame):
    """Extract main numbers (6), optional bonus, and optional dates. Validate 1..49."""
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in main_cols):
        return None, None, None

    # Main numbers
    numbers_df = df[main_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if not numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    # Bonus number (keep all rows; allow NaN)
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors="coerce")
        bonus_series = bonus_series.where(bonus_series.between(1, 49))  # invalid numbers become NaN

    # Flexible date parsing
    date_col = next((col for col in ["DATE", "Draw Date", "Draw_Date", "Date"] if col in df.columns), None)
    dates = None
    if date_col:
        import re
        tmp = df[date_col].astype(str).apply(lambda x: re.sub(r"(\d+)(st|nd|rd|th)", r"\1", x))
        tmp = pd.to_datetime(tmp, errors="coerce")
        dates = tmp

    return numbers_df.astype(int), bonus_series, dates  # return bonus_series as-is

@st.cache_data
def compute_frequencies(numbers_df: pd.DataFrame):
    return Counter(numbers_df.values.flatten())

@st.cache_data
def compute_pair_frequencies(numbers_df: pd.DataFrame, limit: int = 500):
    counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        counts.update(combinations(sorted(row.values), 2))
    return counts

@st.cache_data
def compute_triplet_frequencies(numbers_df: pd.DataFrame, limit: int = 500):
    counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        counts.update(combinations(sorted(row.values), 3))
    return counts

def compute_number_gaps(numbers_df: pd.DataFrame, dates: pd.Series | None = None):
    last_seen = {n: -1 for n in range(1, 50)}
    df = numbers_df.copy()
    if dates is not None and len(dates) == len(numbers_df):
        df = df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    total_draws = len(df)
    return {n: (total_draws - 1 - last_seen[n]) if last_seen[n] != -1 else total_draws for n in range(1, 50)}

def most_common_per_position(numbers_df: pd.DataFrame):
    result = {}
    for col in numbers_df.columns:
        num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (int(num), int(freq))
    return result

# -------------
# Generators
# -------------

def generate_ticket(pool: list[int]):
    pool = [int(n) for n in pool]
    if len(pool) < 6:
        pool = [n for n in range(1, 50)]
    return sorted(random.sample(pool, 6))

def generate_balanced_ticket():
    """3 odd / 3 even, 2 from each zone, sum in [100, 180]."""
    while True:
        ticket = []
        ticket += random.sample(range(1, 17), 2)
        ticket += random.sample(range(17, 34), 2)
        ticket += random.sample(range(34, 50), 2)
        ticket = sorted(ticket)
        odds = sum(1 for n in ticket if n % 2 == 1)
        total = sum(ticket)
        if odds == 3 and 100 <= total <= 180:
            return ticket

@st.cache_data
def compute_delta_distribution(numbers_df: pd.DataFrame):
    deltas = []
    for row in numbers_df.values:
        row = sorted(row)
        deltas.extend([row[i + 1] - row[i] for i in range(5)])
    return Counter(deltas)

def generate_delta_ticket(delta_counter: Counter):
    top_deltas = [d for d, _ in delta_counter.most_common(10)] or [1,2,3,4,5]
    for _ in range(200):
        start = random.randint(1, 20)
        seq = [start]
        for _ in range(5):
            d = random.choice(top_deltas)
            seq.append(seq[-1] + d)
        seq = [n for n in seq if 1 <= n <= 49]
        if len(seq) == 6:
            return sorted(seq)
    return sorted(random.sample(range(1, 50), 6))

def generate_zone_ticket(mode: str = "3-zone"):
    if mode == "3-zone":
        low = random.sample(range(1, 17), 2)
        mid = random.sample(range(17, 34), 2)
        high = random.sample(range(34, 50), 2)
        return sorted(low + mid + high)
    else:
        q1 = random.sample(range(1, 13), 1)
        q2 = random.sample(range(13, 25), 2)
        q3 = random.sample(range(25, 37), 2)
        q4 = random.sample(range(37, 50), 1)
        return sorted(q1 + q2 + q3 + q4)

def passes_constraints(ticket: list[int], sum_min, sum_max, spread_min, spread_max, odd_count=None):
    total = sum(ticket)
    spread = max(ticket) - min(ticket)
    odds = sum(1 for n in ticket if n % 2 == 1)
    if odd_count is not None and odds != odd_count:
        return False
    if not (sum_min <= total <= sum_max):
        return False
    if not (spread_min <= spread <= spread_max):
        return False
    return True

def apply_exclusions_to_pool(pool: list[int], excluded: set[int]):
    pool = [n for n in pool if n not in excluded]
    if len(pool) < 6:
        pool = [n for n in range(1,50) if n not in excluded]
    return pool

@st.cache_data
def compute_repeat_frequency(numbers_df: pd.DataFrame):
    past = [set(row) for row in numbers_df.values.tolist()]
    repeats = Counter()
    for i in range(1, len(past)):
        for n in past[i].intersection(past[i - 1]):
            repeats[n] += 1
    return repeats

def generate_repeat_ticket(last_draw: set[int], excluded: set[int], repeat_count: int = 1):
    candidates = list(last_draw - excluded)
    if len(candidates) < repeat_count:
        repeat_count = max(0, len(candidates))
    chosen_repeats = random.sample(candidates, repeat_count) if repeat_count > 0 else []
    pool = [n for n in range(1, 50) if n not in set(chosen_repeats) | excluded]
    rest = random.sample(pool, 6 - len(chosen_repeats))
    return sorted(chosen_repeats + rest)

def try_generate_with_constraints(gen_callable, *, sum_min, sum_max, spread_min, spread_max, odd_count, max_tries=200):
    last_ticket = None
    for _ in range(max_tries):
        t = gen_callable()
        last_ticket = t
        if passes_constraints(t, sum_min, sum_max, spread_min, spread_max, odd_count):
            return t
    return last_ticket

# ======================
# CSV Upload & Cleaning
# ======================
uploaded_file = st.file_uploader(
    "📂 Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV must include columns NUMBER DRAWN 1–6. Optional: BONUS NUMBER, DATE.",
)

if not uploaded_file:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
    st.stop()

try:
    # Read CSV
    raw_df = pd.read_csv(uploaded_file)
    
    # Extract numbers, bonus, dates
    numbers_df, bonus_series, dates = extract_numbers_and_bonus(raw_df)
    
    if numbers_df is None:
        st.error("❌ Invalid CSV. Ensure columns NUMBER DRAWN 1–6 exist with values between 1 and 49.")
        st.stop()
    
    # Align lengths in case of missing rows
    min_len = len(numbers_df)
    if bonus_series is not None:
        bonus_series = bonus_series[:min_len].reset_index(drop=True)
    if dates is not None:
        dates = dates[:min_len].reset_index(drop=True)
    
    # Combine into display DataFrame
    display_df = numbers_df.copy().reset_index(drop=True)
    if bonus_series is not None:
        display_df["BONUS NUMBER"] = bonus_series.astype("Int64")
    if dates is not None:
        display_df["DATE"] = dates.astype(str)
    
    # Sort by date if available (ignore index to avoid misalignment)
    if "DATE" in display_df.columns:
        display_df = display_df.sort_values("DATE", ignore_index=True)
        # Update numbers_df and bonus_series to match display_df order
        numbers_df = display_df[numbers_df.columns]
        if bonus_series is not None:
            bonus_series = display_df["BONUS NUMBER"]
    
    st.subheader(f"✅ Uploaded Data ({len(display_df)} draws)")
    st.dataframe(display_df)

except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()
# ======================
# Sidebar Controls
# ======================
st.sidebar.header("⚙️ Global Controls")
max_draws = len(numbers_df)
draw_limit = st.sidebar.slider("Number of past draws to analyze", min_value=10, max_value=max_draws, value=max_draws)
numbers_df = numbers_df.tail(draw_limit).reset_index(drop=True)
num_tickets = st.sidebar.slider("Tickets to generate (per tab)", 1, 12, 6)
excluded_str = st.sidebar.text_input("Exclude numbers (comma-separated)", "")
excluded = {int(x.strip()) for x in excluded_str.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 49}
sum_min, sum_max = st.sidebar.slider("Sum range", 60, 250, (100,180))
spread_min, spread_max = st.sidebar.slider("Spread range (max - min)", 5, 48, (10,40))
odd_mode = st.sidebar.selectbox("Odd/Even constraint", ["Any","Exactly 0 odd","1","2","3","4","5","6"])
odd_count = None if odd_mode=="Any" else int(odd_mode.split()[0]) if "Exactly" not in odd_mode else int(odd_mode.split()[2])

# ======================
# Precompute analytics
# ======================
counter = compute_frequencies(numbers_df)
hot = [int(n) for n,_ in counter.most_common(6)]
cold = [int(n) for n,_ in counter.most_common()[:-7:-1]]
delta_counter = compute_delta_distribution(numbers_df)
last_draw = set(numbers_df.iloc[-1])
repeats = compute_repeat_frequency(numbers_df)
pos_common = most_common_per_position(numbers_df)
gaps = compute_number_gaps(numbers_df)

# ======================
# Tabs
# ======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Hot / Cold / Overdue",
    "Δ Delta System",
    "Cluster / Zone Coverage",
    "Sum & Spread Filters",
    "Smart Exclusion",
    "Repeat Hit Analysis",
    "Jackpot Simulation",
    "ML Prediction",
])

# -------------------------
# Tab 1: Hot / Cold / Overdue
# -------------------------
with tab1:
    st.subheader("🔥 Hot / ❄️ Cold / ⏳ Overdue")
    overdue_threshold = st.slider("Highlight numbers very overdue", 0, max(gaps.values()), int(np.quantile(list(gaps.values()),0.75)))
    overdue_numbers = [n for n,g in gaps.items() if g >= overdue_threshold]
    choice = st.radio("Bias strategy", ["Hot","Cold","Overdue"], horizontal=True)
    tickets = []
    pool_base = list(range(1,50))
    for _ in range(num_tickets):
        if choice=="Hot":
            pool = hot + [n for n in pool_base if n not in hot]
        elif choice=="Cold":
            pool = cold + [n for n in pool_base if n not in cold]
        else:
            pool = overdue_numbers + [n for n in pool_base if n not in overdue_numbers]
        pool = apply_exclusions_to_pool(pool, excluded)
        ticket = try_generate_with_constraints(lambda: generate_ticket(pool), sum_min=sum_min, sum_max=sum_max, spread_min=spread_min, spread_max=spread_max, odd_count=odd_count)
        tickets.append(ticket)
    st.dataframe(pd.DataFrame(tickets, columns=[f"N{i}" for i in range(1,7)]))

# -------------------------
# Tab 2: Δ Delta System
# -------------------------
with tab2:
    st.subheader("Δ Delta System Tickets")
    tickets = [try_generate_with_constraints(lambda: generate_delta_ticket(delta_counter), sum_min=sum_min, sum_max=sum_max, spread_min=spread_min, spread_max=spread_max, odd_count=odd_count) for _ in range(num_tickets)]
    st.dataframe(pd.DataFrame(tickets, columns=[f"N{i}" for i in range(1,7)]))

# -------------------------
# Tab 3: Cluster / Zone Coverage
# -------------------------
with tab3:
    st.subheader("Cluster / Zone Tickets")
    tickets = [try_generate_with_constraints(lambda: generate_zone_ticket("3-zone"), sum_min=sum_min, sum_max=sum_max, spread_min=spread_min, spread_max=spread_max, odd_count=odd_count) for _ in range(num_tickets)]
    st.dataframe(pd.DataFrame(tickets, columns=[f"N{i}" for i in range(1,7)]))

# -------------------------
# Tab 4: Sum & Spread Filters
# -------------------------
with tab4:
    st.subheader("Tickets respecting Sum & Spread filters")
    tickets = [try_generate_with_constraints(generate_balanced_ticket, sum_min=sum_min, sum_max=sum_max, spread_min=spread_min, spread_max=spread_max, odd_count=odd_count) for _ in range(num_tickets)]
    st.dataframe(pd.DataFrame(tickets, columns=[f"N{i}" for i in range(1,7)]))

# -------------------------
# Tab 5: Smart Exclusion
# -------------------------
with tab5:
    st.subheader("Tickets avoiding excluded numbers")
    tickets = [try_generate_with_constraints(lambda: generate_ticket(apply_exclusions_to_pool(list(range(1,50)), excluded)), sum_min=sum_min, sum_max=sum_max, spread_min=spread_min, spread_max=spread_max, odd_count=odd_count) for _ in range(num_tickets)]
    st.dataframe(pd.DataFrame(tickets, columns=[f"N{i}" for i in range(1,7)]))

# -------------------------
# Tab 6: Repeat Hit Analysis
# -------------------------
with tab6:
    st.subheader("Tickets with repeats from last draw")
    tickets = [generate_repeat_ticket(last_draw, excluded, repeat_count=1) for _ in range(num_tickets)]
    st.dataframe(pd.DataFrame(tickets, columns=[f"N{i}" for i in range(1,7)]))
    st.write("Numbers repeated from last draw:", last_draw)

# -------------------------
# Tab 7: Jackpot Simulation
# -------------------------
with tab7:
    st.subheader("Jackpot Simulation")
    test_ticket = st.multiselect("Select numbers to test", options=list(range(1,50)))
    if len(test_ticket) == 6:
        matches = numbers_df.apply(lambda row: len(set(row.values) & set(test_ticket)), axis=1)
        st.write("Match distribution for your ticket:")
        st.bar_chart(matches.value_counts().sort_index())
    else:
        st.info("Select exactly 6 numbers to simulate.")

# -------------------------
# Tab 8: ML Prediction
# -------------------------
with tab8:
    st.subheader("ML-based Prediction")
    n_future = st.slider("Predict next N draws", 1, 5, 1)
    X = numbers_df.shift(1).fillna(0).astype(int)
    y = numbers_df.astype(int)
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
    clf.fit(X, y)
    last_row = X.iloc[-1].values.reshape(1, -1)
    predictions = []
    for _ in range(n_future):
        pred = clf.predict(last_row)[0]
        predictions.append(sorted(pred))
        last_row = pred.reshape(1, -1)
    st.dataframe(pd.DataFrame(predictions, columns=[f"N{i}" for i in range(1,7)]))

# ======================
# CSV Export
# ======================
st.download_button(
    label="💾 Export analyzed data CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="lotto_analyzed.csv",
    mime="text/csv",
)
