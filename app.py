import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
import random
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import re

# ======================
# Streamlit Page Config
# ======================
st.set_page_config(
    page_title="🎲 Canada Lotto 6/49 Analyzer",
    page_icon="🎲",
    layout="wide",
)
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and backtest strategies.")

# ======================
# Helper Functions
# ======================
def extract_numbers_and_bonus(df: pd.DataFrame):
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in main_cols):
        return None, None, None

    numbers_df = df[main_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if not numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors="coerce")
        bonus_series = bonus_series.where(bonus_series.between(1, 49))

    date_col = next((col for col in ["DATE", "Draw Date", "Draw_Date", "Date"] if col in df.columns), None)
    dates = None
    if date_col:
        tmp = df[date_col].astype(str).apply(lambda x: re.sub(r"(\d+)(st|nd|rd|th)", r"\1", x))
        tmp = pd.to_datetime(tmp, errors="coerce")
        dates = tmp

    return numbers_df.astype(int), bonus_series, dates

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

# -----------------------------
# Ticket Generators
# -----------------------------
def generate_ticket(pool: list[int]):
    pool = [int(n) for n in pool]
    if len(pool) < 6:
        pool = [n for n in range(1, 50)]
    return sorted(random.sample(pool, 6))

def generate_balanced_ticket():
    while True:
        ticket = random.sample(range(1, 17), 2) + random.sample(range(17, 34), 2) + random.sample(range(34, 50), 2)
        ticket = sorted(ticket)
        if sum(1 for n in ticket if n % 2 == 1) == 3 and 100 <= sum(ticket) <= 180:
            return ticket

@st.cache_data
def compute_delta_distribution(numbers_df: pd.DataFrame):
    deltas = []
    for row in numbers_df.values:
        row = sorted(row)
        deltas.extend([row[i + 1] - row[i] for i in range(5)])
    return Counter(deltas)

def generate_delta_ticket(delta_counter: Counter):
    top_deltas = [d for d, _ in delta_counter.most_common(10)] or [1, 2, 3, 4, 5]
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
        return sorted(random.sample(range(1, 17), 2) + random.sample(range(17, 34), 2) + random.sample(range(34, 50), 2))
    return sorted(random.sample(range(1, 13), 1) + random.sample(range(13, 25), 2) + random.sample(range(25, 37), 2) + random.sample(range(37, 50), 1))

def passes_constraints(ticket: list[int], sum_min: int, sum_max: int, spread_min: int, spread_max: int, odd_count: int | None):
    total, spread, odds = sum(ticket), max(ticket)-min(ticket), sum(1 for n in ticket if n % 2 == 1)
    if odd_count is not None and odds != odd_count: return False
    return sum_min <= total <= sum_max and spread_min <= spread <= spread_max

def apply_exclusions_to_pool(pool: list[int], excluded: set[int]):
    pool = [n for n in pool if n not in excluded]
    if len(pool) < 6:
        pool = [n for n in range(1, 50) if n not in excluded]
    return pool

@st.cache_data
def compute_repeat_frequency(numbers_df: pd.DataFrame):
    past = [set(row) for row in numbers_df.values.tolist()]
    repeats = Counter()
    for i in range(1, len(past)):
        for n in past[i].intersection(past[i-1]):
            repeats[n] += 1
    return repeats

def generate_repeat_ticket(last_draw: set[int], excluded: set[int], repeat_count: int = 1):
    candidates = list(last_draw - excluded)
    repeat_count = min(repeat_count, len(candidates))
    chosen_repeats = random.sample(candidates, repeat_count) if repeat_count > 0 else []
    pool = [n for n in range(1, 50) if n not in set(chosen_repeats) | excluded]
    return sorted(chosen_repeats + random.sample(pool, 6 - len(chosen_repeats)))

def simulate_strategy(strategy_func, numbers_df: pd.DataFrame, n: int = 1000):
    past_draws = [set(row) for row in numbers_df.values.tolist()]
    results = {3: 0, 4: 0, 5: 0, 6: 0}
    for _ in range(n):
        ticket = set(strategy_func())
        for draw in past_draws:
            hits = len(ticket.intersection(draw))
            if hits >= 3: results[hits] += 1
    return results

def try_generate_with_constraints(gen_callable, *, sum_min, sum_max, spread_min, spread_max, odd_count, max_tries: int = 200):
    last_ticket = None
    for _ in range(max_tries):
        t = gen_callable()
        last_ticket = t
        if passes_constraints(t, sum_min, sum_max, spread_min, spread_max, odd_count):
            return t
    return last_ticket

# ======================
# File Upload & Parsing
# ======================
uploaded_file = st.file_uploader("📂 Upload a Lotto 6/49 CSV file", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
    numbers_df, bonus_series, dates = extract_numbers_and_bonus(raw_df)
    if numbers_df is None:
        st.error("❌ Invalid CSV. Ensure NUMBER DRAWN 1–6 exist with values 1–49.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

# -----------------------------
# Display Uploaded Data
# -----------------------------
display_df = numbers_df.copy()
if bonus_series is not None: display_df["BONUS NUMBER"] = bonus_series.astype("Int64").values
if dates is not None: display_df["DATE"] = dates.astype(str).values
st.subheader(f"✅ Uploaded Data ({len(numbers_df)} draws)")
st.dataframe(display_df)

# ======================
# Analytics & Generators
# ======================
counter = compute_frequencies(numbers_df)
hot = [int(n) for n,_ in counter.most_common(6)]
cold = [int(n) for n,_ in counter.most_common()[:-7:-1]]
delta_counter = compute_delta_distribution(numbers_df)
last_draw = set(numbers_df.iloc[-1])
repeats = compute_repeat_frequency(numbers_df)

st.markdown("### 🔥 Hot & ❄️ Cold Numbers")
st.write(f"**Hot:** {hot}, **Cold:** {cold}")

# -----------------------------
# ML Prediction
# -----------------------------
number_cols = [f"NUMBER DRAWN {i}" for i in range(1,7)]

def compute_features(draw):
    numbers = [int(n) for n in draw[:6]]
    deltas = [numbers[i+1]-numbers[i] for i in range(5)]
    odd_count = sum([n%2 for n in numbers])
    even_count = 6 - odd_count
    total_sum = sum(numbers)
    return deltas + [odd_count, even_count, total_sum]

feature_list = raw_df[number_cols].apply(compute_features, axis=1)
feature_df = pd.DataFrame(feature_list.tolist(), columns=[f"delta_{i}" for i in range(1,6)] + ["odd_count","even_count","sum_numbers"])
targets = raw_df[number_cols].astype(int)

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
model.fit(feature_df, targets)
st.success("✅ ML Model trained.")

last_features = np.array(compute_features(raw_df.iloc[-1][number_cols])).reshape(1,-1)
preds = model.predict(last_features)
suggested_tickets = []
for _ in range(5):
    ticket = np.clip(preds[0] + np.random.randint(-2,3,6), 1, 49)
    ticket = sorted(list(set(int(n) for n in ticket)))
    while len(ticket)<6:
        n=random.randint(1,49)
        if n not in ticket: ticket.append(n)
    suggested_tickets.append(sorted(ticket))
st.subheader("🎟 ML-Based Ticket Suggestions")
for i, t in enumerate(suggested_tickets,1): st.write(f"Ticket {i}: {t}")

st.markdown("**Disclaimer:** Lotto 6/49 is a game of chance. These analyses cannot predict future draws.")
