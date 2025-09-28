import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
import random
import plotly.express as px
import numpy as np
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
st.write("Analyze historical draws, identify patterns, generate tickets, and backtest strategies.")

# ======================
# Helper Functions
# ======================
def extract_numbers_and_bonus(df: pd.DataFrame):
    """Extract main numbers (6), optional bonus, and optional dates."""
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
        import re
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

def generate_ticket(pool: list[int]):
    pool = [int(n) for n in pool]
    if len(pool) < 6:
        pool = [n for n in range(1, 50)]
    return sorted(random.sample(pool, 6))

def generate_balanced_ticket():
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
    top_deltas = [int(d) for d, _ in delta_counter.most_common(10)] or [1, 2, 3, 4, 5]
    for _ in range(200):
        start = random.randint(1, 20)
        seq = [start]
        for _ in range(5):
            d = int(random.choice(top_deltas))
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
    q1 = random.sample(range(1, 13), 1)
    q2 = random.sample(range(13, 25), 2)
    q3 = random.sample(range(25, 37), 2)
    q4 = random.sample(range(37, 50), 1)
    return sorted(q1 + q2 + q3 + q4)

def passes_constraints(ticket: list[int], sum_min: int, sum_max: int, spread_min: int, spread_max: int, odd_count: int | None):
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
        pool = [n for n in range(1, 50) if n not in excluded]
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

def try_generate_with_constraints(gen_callable, *, sum_min, sum_max, spread_min, spread_max, odd_count, max_tries: int = 200):
    last_ticket = None
    for _ in range(max_tries):
        t = gen_callable()
        last_ticket = t
        if passes_constraints(t, sum_min, sum_max, spread_min, spread_max, odd_count):
            return t
    return last_ticket

# ======================
# Upload CSV
# ======================
uploaded_file = st.file_uploader("📂 Upload Lotto 6/49 CSV", type=["csv"])
if not uploaded_file:
    st.info("Upload CSV with columns NUMBER DRAWN 1–6 (optional: BONUS NUMBER, DATE).")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
numbers_df, bonus_series, dates = extract_numbers_and_bonus(df_raw)
if numbers_df is None:
    st.error("Invalid CSV. Ensure NUMBER DRAWN 1–6 exist and are 1–49.")
    st.stop()

display_df = numbers_df.copy()
if bonus_series is not None and len(bonus_series) == len(display_df):
    display_df["BONUS NUMBER"] = bonus_series.reset_index(drop=True).astype("Int64")
if dates is not None and len(dates) == len(display_df):
    display_df["DATE"] = dates.reset_index(drop=True).astype(str)

st.subheader("✅ Uploaded Draws")
st.dataframe(display_df)

# ======================
# Sidebar controls
# ======================
st.sidebar.header("⚙️ Global Controls")
draw_limit = st.sidebar.slider("Number of past draws to analyze", 10, len(numbers_df), len(numbers_df))
numbers_df = numbers_df.tail(draw_limit).reset_index(drop=True)
num_tickets = st.sidebar.slider("Tickets to generate per tab", 1, 12, 6)
excluded_str = st.sidebar.text_input("Exclude numbers (comma-separated)", "")
excluded = {int(x.strip()) for x in excluded_str.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 49}
sum_min, sum_max = st.sidebar.slider("Sum range", 60, 250, (100, 180))
spread_min, spread_max = st.sidebar.slider("Spread range (max-min)", 5, 48, (10, 40))
odd_mode = st.sidebar.selectbox("Odd/Even constraint", ["Any", "Exactly 0 odd", "1", "2", "3", "4", "5", "6"])
odd_count = None if odd_mode == "Any" else int(odd_mode.split()[0]) if odd_mode.startswith("Exactly") else int(odd_mode)

# ======================
# Analytics
# ======================
counter = compute_frequencies(numbers_df)
hot = [int(n) for n, _ in counter.most_common(6)]
cold = [int(n) for n, _ in counter.most_common()[:-7:-1]]
delta_counter = compute_delta_distribution(numbers_df)
last_draw_set = set(numbers_df.iloc[-1])
repeats = compute_repeat_frequency(numbers_df)
gaps = compute_number_gaps(numbers_df)

# ======================
# Tabs
# ======================
tabs = st.tabs([
    "Hot / Cold / Overdue",
    "Δ Delta System",
    "Cluster / Zone Coverage",
    "Sum & Spread Filters",
    "Smart Exclusion",
    "Repeat Hit Analysis",
    "Jackpot Simulation",
    "ML-Based Predictions"
])

# ------- Tab 1: Hot / Cold / Overdue -------
with tabs[0]:
    st.subheader("🔥 Hot / ❄️ Cold / ⏳ Overdue")
    overdue_threshold = st.slider("Highlight numbers very overdue", 0, max(gaps.values()), int(np.quantile(list(gaps.values()), 0.75)))
    overdue_numbers = [n for n, g in gaps.items() if g >= overdue_threshold]
    choice = st.radio("Bias strategy", ["Hot", "Cold", "Overdue"], horizontal=True)
    tickets = []
    pool_base = list(range(1,50))
    for _ in range(num_tickets):
        if choice=="Hot": pool = hot + [n for n in pool_base if n not in hot]
        elif choice=="Cold": pool = cold + [n for n in pool_base if n not in cold]
        else: pool = overdue_numbers + [n for n in pool_base if n not in overdue_numbers]
        pool = apply_exclusions_to_pool(pool, excluded)
        ticket = try_generate_with_constraints(lambda: generate_ticket(pool), sum_min=sum_min, sum_max=sum_max,
                                               spread_min=spread_min, spread_max=spread_max, odd_count=odd_count)
        tickets.append(ticket)
    for i, t in enumerate(tickets,1): st.write(f"Ticket {i}: {t}")

# ------- Tab 2: Δ Delta System -------
with tabs[1]:
    st.subheader("Δ Delta System")
    tickets = [try_generate_with_constraints(lambda: generate_delta_ticket(delta_counter),
                                             sum_min=sum_min, sum_max=sum_max,
                                             spread_min=spread_min, spread_max=spread_max,
                                             odd_count=odd_count) for _ in range(num_tickets)]
    for i, t in enumerate(tickets,1): st.write(f"Ticket {i}: {t}")

# ------- Tab 3: Cluster / Zone Coverage -------
with tabs[2]:
    st.subheader("Cluster / Zone Coverage")
    zone_mode = st.selectbox("Zone mode", ["3-zone", "4-zone"])
    tickets = [try_generate_with_constraints(lambda: generate_zone_ticket(zone_mode),
                                             sum_min=sum_min, sum_max=sum_max,
                                             spread_min=spread_min, spread_max=spread_max,
                                             odd_count=odd_count) for _ in range(num_tickets)]
    for i, t in enumerate(tickets,1): st.write(f"Ticket {i}: {t}")

# ------- Tab 4: Sum & Spread Filters -------
with tabs[3]:
    st.subheader("Sum & Spread Filters")
    tickets = [try_generate_with_constraints(generate_balanced_ticket,
                                             sum_min=sum_min, sum_max=sum_max,
                                             spread_min=spread_min, spread_max=spread_max,
                                             odd_count=odd_count) for _ in range(num_tickets)]
    for i, t in enumerate(tickets,1): st.write(f"Ticket {i}: {t}")

# ------- Tab 5: Smart Exclusion -------
with tabs[4]:
    st.subheader("Smart Exclusion")
    tickets = [try_generate_with_constraints(lambda: generate_ticket([n for n in range(1,50) if n not in excluded]),
                                             sum_min=sum_min, sum_max=sum_max,
                                             spread_min=spread_min, spread_max=spread_max,
                                             odd_count=odd_count) for _ in range(num_tickets)]
    for i, t in enumerate(tickets,1): st.write(f"Ticket {i}: {t}")

# ------- Tab 6: Repeat Hit Analysis -------
with tabs[5]:
    st.subheader("Repeat Hit Analysis")
    repeat_count = st.slider("Max repeats from last draw", 0, 6, 2)
    tickets = [try_generate_with_constraints(lambda: generate_repeat_ticket(last_draw_set, excluded, repeat_count),
                                             sum_min=sum_min, sum_max=sum_max,
                                             spread_min=spread_min, spread_max=spread_max,
                                             odd_count=odd_count) for _ in range(num_tickets)]
    for i, t in enumerate(tickets,1): st.write(f"Ticket {i}: {t}")

# ------- Tab 7: Jackpot Simulation -------
with tabs[6]:
    st.subheader("Jackpot Simulation")
    st.info("Simulate random draws and see if you hit the jackpot")
    simulate_draws = st.number_input("Number of random draws to simulate", min_value=1, max_value=1000, value=100)
    jackpot_hits = 0
    for _ in range(simulate_draws):
        draw = sorted(random.sample(range(1,50),6))
        for t in [generate_ticket(range(1,50)) for _ in range(num_tickets)]:
            if set(draw) == set(t):
                jackpot_hits += 1
    st.write(f"Jackpot hits: {jackpot_hits} / {simulate_draws*num_tickets}")

# ------- Tab 8: ML-Based Predictions -------
with tabs[7]:
    st.subheader("ML-Based Predictions")
    X = numbers_df[:-1].values
    y = numbers_df[1:].values
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    clf.fit(X, y)
    prediction = clf.predict(numbers_df.tail(1).values)[0]
    st.write("Predicted next draw:", sorted([int(n) for n in prediction]))
