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
st.write("Analyze historical draws, generate tickets, and explore ML predictions.")

# ======================
# Helper Functions
# ======================

def extract_numbers_and_bonus(df):
    """Extract main numbers and optional bonus numbers."""
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

    # Flexible date parsing
    date_col = next((col for col in ["DATE", "Draw Date", "Draw_Date", "Date"] if col in df.columns), None)
    dates = None
    if date_col:
        import re
        tmp = df[date_col].astype(str).apply(lambda x: re.sub(r"(\d+)(st|nd|rd|th)", r"\1", x))
        tmp = pd.to_datetime(tmp, errors="coerce")
        dates = tmp

    return numbers_df.astype(int), bonus_series, dates

@st.cache_data
def compute_frequencies(numbers_df):
    return Counter(numbers_df.values.flatten())

@st.cache_data
def compute_pair_frequencies(numbers_df, limit=500):
    counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        counts.update(combinations(sorted(row.values), 2))
    return counts

@st.cache_data
def compute_triplet_frequencies(numbers_df, limit=500):
    counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        counts.update(combinations(sorted(row.values), 3))
    return counts

def most_common_per_position(numbers_df):
    result = {}
    for col in numbers_df.columns:
        num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (int(num), int(freq))
    return result

# -----------------
# Ticket Generators
# -----------------
def generate_ticket(pool):
    pool = [int(n) for n in pool]
    if len(pool) < 6:
        pool = list(range(1, 50))
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

def generate_delta_ticket(delta_counter=None):
    top_deltas = [d for d, _ in (delta_counter.most_common(10) if delta_counter else [(1,1),(2,1),(3,1),(4,1),(5,1)])]
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

def generate_zone_ticket(mode="3-zone"):
    if mode == "3-zone":
        low = random.sample(range(1, 17), 2)
        mid = random.sample(range(17, 34), 2)
        high = random.sample(range(34, 50), 2)
        return sorted(low + mid + high)
    # Quartiles
    q1 = random.sample(range(1, 13), 1)
    q2 = random.sample(range(13, 25), 2)
    q3 = random.sample(range(25, 37), 2)
    q4 = random.sample(range(37, 50), 1)
    return sorted(q1 + q2 + q3 + q4)

def generate_smart_exclusion_ticket(excluded=set()):
    pool = [n for n in range(1, 50) if n not in excluded]
    return sorted(random.sample(pool, 6))

def generate_repeat_ticket(last_draw=set(), excluded=set(), repeat_count=1):
    candidates = list(last_draw - excluded)
    chosen_repeats = random.sample(candidates, min(len(candidates), repeat_count))
    pool = [n for n in range(1, 50) if n not in set(chosen_repeats) | excluded]
    rest = random.sample(pool, 6 - len(chosen_repeats))
    return sorted(chosen_repeats + rest)

def simulate_jackpot_ticket():
    return sorted(random.sample(range(1, 50), 6))

# ======================
# File Upload
# ======================
uploaded_file = st.file_uploader("📂 Upload Lotto 6/49 CSV", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV with columns NUMBER DRAWN 1-6")
    st.stop()

df = pd.read_csv(uploaded_file)
numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
if numbers_df is None:
    st.error("Invalid CSV. Ensure columns NUMBER DRAWN 1–6 exist with values 1–49.")
    st.stop()

st.subheader("📄 Historical Draws Preview")
display_df = numbers_df.copy()
if bonus_series is not None and len(bonus_series) == len(display_df):
    display_df["BONUS NUMBER"] = bonus_series.astype("Int64").values
if dates is not None and len(dates) == len(display_df):
    display_df["DATE"] = dates.dt.strftime("%Y-%m-%d").values
st.dataframe(display_df.tail(10))

# ======================
# Tab Layout
# ======================
num_tickets = st.sidebar.slider("Tickets per generator", 1, 12, 6)
excluded_str = st.sidebar.text_input("Exclude numbers (comma-separated)", "")
excluded = {int(x.strip()) for x in excluded_str.split(",") if x.strip().isdigit()}

delta_counter = compute_pair_frequencies(numbers_df)  # use pair freq as delta approx
last_draw_set = set(numbers_df.iloc[-1])

tabs = st.tabs([
    "Hot / Cold / Overdue",
    "Δ Delta System",
    "Cluster / Zone Coverage",
    "Sum & Spread Filters",
    "Smart Exclusion",
    "Repeat Hit Analysis",
    "ML Prediction",
])

# Tab 1: Hot / Cold / Overdue
with tabs[0]:
    st.subheader("🔥 Hot / ❄️ Cold / ⏳ Overdue")
    counter = compute_frequencies(numbers_df)
    hot = [int(n) for n,_ in counter.most_common(6)]
    cold = [int(n) for n,_ in counter.most_common()[:-7:-1]]
    overdue = [int(n) for n in range(1,50) if n not in numbers_df.values.flatten()]
    choice = st.radio("Bias strategy", ["Hot", "Cold", "Overdue"])
    tickets = []
    pool_base = list(range(1,50))
    for _ in range(num_tickets):
        pool = hot if choice=="Hot" else cold if choice=="Cold" else overdue
        pool += [n for n in pool_base if n not in pool]
        ticket = generate_ticket([n for n in pool if n not in excluded])
        tickets.append(ticket)
    for i,t in enumerate(tickets,1):
        st.write(f"Ticket {i}: {[int(n) for n in t]}")

# Tab 2: Delta System
with tabs[1]:
    st.subheader("Δ Delta System Tickets")
    tickets = [generate_delta_ticket(delta_counter) for _ in range(num_tickets)]
    for i,t in enumerate(tickets,1):
        st.write(f"Ticket {i}: {[int(n) for n in t]}")

# Tab 3: Cluster / Zone Coverage
with tabs[2]:
    st.subheader("📊 Cluster / Zone Coverage Tickets")
    tickets = [generate_zone_ticket() for _ in range(num_tickets)]
    for i,t in enumerate(tickets,1):
        st.write(f"Ticket {i}: {[int(n) for n in t]}")

# Tab 4: Sum & Spread Filters
with tabs[3]:
    st.subheader("➕ Sum & Spread Filtered Tickets")
    tickets = [generate_balanced_ticket() for _ in range(num_tickets)]
    for i,t in enumerate(tickets,1):
        st.write(f"Ticket {i}: {[int(n) for n in t]}")

# Tab 5: Smart Exclusion
with tabs[4]:
    st.subheader("🚫 Smart Exclusion Tickets")
    tickets = [generate_smart_exclusion_ticket(excluded) for _ in range(num_tickets)]
    for i,t in enumerate(tickets,1):
        st.write(f"Ticket {i}: {[int(n) for n in t]}")

# Tab 6: Repeat Hit Analysis
with tabs[5]:
    st.subheader("🔁 Repeat Hit Tickets")
    repeat_count = st.slider("Numbers to repeat from last draw", 0, 2, 1)
    tickets = [generate_repeat_ticket(last_draw_set, excluded, repeat_count) for _ in range(num_tickets)]
    for i,t in enumerate(tickets,1):
        st.write(f"Ticket {i}: {[int(n) for n in t]}")

# Tab 7: ML Prediction
with tabs[6]:
    st.subheader("🤖 ML-Based Ticket Prediction")
    number_cols = [f"NUMBER DRAWN {i}" for i in range(1,7)]

    def compute_features(draw):
        numbers = [int(n) for n in draw[:6]]
        deltas = [numbers[i+1]-numbers[i] for i in range(5)]
        odd_count = sum([n%2 for n in numbers])
        even_count = 6-odd_count
        total_sum = sum(numbers)
        return deltas + [odd_count, even_count, total_sum]

    feature_list = df[number_cols].apply(compute_features, axis=1)
    feature_df = pd.DataFrame(feature_list.tolist(),
                              columns=[f"delta_{i}" for i in range(1,6)] + ["odd_count","even_count","sum_numbers"])
    targets = df[number_cols].astype(int)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    model.fit(feature_df, targets)
    st.success("✅ ML Model trained on historical draws.")

    last_draw_features = np.array(compute_features(df.iloc[-1][number_cols])).reshape(1,-1)
    preds = model.predict(last_draw_features)

    tickets = []
    for _ in range(num_tickets):
        ticket = np.clip(preds[0] + np.random.randint(-2,3,6),1,49)
        ticket = sorted(list({int(n) for n in ticket}))
        while len(ticket)<6:
            n=random.randint(1,49)
            if n not in ticket: ticket.append(n)
        tickets.append(sorted(ticket))
    for i,t in enumerate(tickets,1):
        st.write(f
