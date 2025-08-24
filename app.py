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
    
    # Main numbers
    numbers_df = df[main_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if not numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    # Bonus
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors="coerce").where(
            df[bonus_col].between(1,49)
        )

    # Dates
    date_col = next((c for c in df.columns if c.lower().replace("_","") in ["drawdate","date"]), None)
    dates = None
    if date_col:
        tmp = df[date_col].astype(str).apply(lambda x: re.sub(r"(\d+)(st|nd|rd|th)", r"\1", x))
        dates = pd.to_datetime(tmp, errors="coerce")
    
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
    last_seen = {n: -1 for n in range(1,50)}
    df = numbers_df.reset_index(drop=True)
    for idx, row in df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    total_draws = len(df)
    return {n: (total_draws - 1 - last_seen[n]) if last_seen[n] != -1 else total_draws for n in range(1,50)}

def most_common_per_position(numbers_df: pd.DataFrame):
    result = {}
    for col in numbers_df.columns:
        num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (int(num), int(freq))
    return result

# ======================
# Generators
# ======================
def generate_ticket(pool: list[int]):
    pool = [int(n) for n in pool]
    if len(pool) < 6:
        pool = list(range(1,50))
    return sorted(random.sample(pool,6))

def generate_balanced_ticket():
    while True:
        ticket = []
        ticket += random.sample(range(1,17),2)
        ticket += random.sample(range(17,34),2)
        ticket += random.sample(range(34,50),2)
        ticket = sorted(ticket)
        odds = sum(1 for n in ticket if n%2==1)
        total = sum(ticket)
        if odds==3 and 100<=total<=180:
            return ticket

@st.cache_data
def compute_delta_distribution(numbers_df: pd.DataFrame):
    deltas = []
    for row in numbers_df.values:
        row = sorted(row)
        deltas.extend([row[i+1]-row[i] for i in range(5)])
    return Counter(deltas)

def generate_delta_ticket(delta_counter: Counter):
    top_deltas = [d for d,_ in delta_counter.most_common(10)] or [1,2,3,4,5]
    for _ in range(200):
        start = random.randint(1,20)
        seq = [start]
        for _ in range(5):
            d = random.choice(top_deltas)
            seq.append(seq[-1]+d)
        seq = [n for n in seq if 1<=n<=49]
        if len(seq)==6:
            return sorted(seq)
    return sorted(random.sample(range(1,50),6))

def generate_zone_ticket(mode:str="3-zone"):
    if mode=="3-zone":
        low = random.sample(range(1,17),2)
        mid = random.sample(range(17,34),2)
        high = random.sample(range(34,50),2)
        return sorted(low+mid+high)
    q1=random.sample(range(1,13),1)
    q2=random.sample(range(13,25),2)
    q3=random.sample(range(25,37),2)
    q4=random.sample(range(37,50),1)
    return sorted(q1+q2+q3+q4)

def passes_constraints(ticket:list[int], sum_min:int, sum_max:int, spread_min:int, spread_max:int, odd_count:int|None):
    total=sum(ticket)
    spread=max(ticket)-min(ticket)
    odds=sum(1 for n in ticket if n%2==1)
    if odd_count is not None and odds!=odd_count: return False
    if not (sum_min<=total<=sum_max): return False
    if not (spread_min<=spread<=spread_max): return False
    return True

def apply_exclusions_to_pool(pool:list[int], excluded:set[int]):
    pool=[n for n in pool if n not in excluded]
    if len(pool)<6: pool=[n for n in range(1,50) if n not in excluded]
    return pool

@st.cache_data
def compute_repeat_frequency(numbers_df: pd.DataFrame):
    past=[set(row) for row in numbers_df.values.tolist()]
    repeats=Counter()
    for i in range(1,len(past)):
        for n in past[i].intersection(past[i-1]):
            repeats[n]+=1
    return repeats

def generate_repeat_ticket(last_draw:set[int], excluded:set[int], repeat_count:int=1):
    candidates=list(last_draw - excluded)
    if len(candidates)<repeat_count:
        repeat_count=max(0,len(candidates))
    chosen=random.sample(candidates,repeat_count) if repeat_count>0 else []
    pool=[n for n in range(1,50) if n not in set(chosen)|excluded]
    rest=random.sample(pool,6-len(chosen))
    return sorted(chosen+rest)

def try_generate_with_constraints(gen_callable, *, sum_min, sum_max, spread_min, spread_max, odd_count, max_tries:int=200):
    last_ticket=None
    for _ in range(max_tries):
        t=gen_callable()
        last_ticket=t
        if passes_constraints(t,sum_min,sum_max,spread_min,spread_max,odd_count):
            return t
    return last_ticket

def simulate_strategy(strategy_func, numbers_df:pd.DataFrame, n:int=1000):
    past_draws=[set(row) for row in numbers_df.values.tolist()]
    results={3:0,4:0,5:0,6:0}
    for _ in range(n):
        ticket=set(strategy_func())
        for draw in past_draws:
            hits=len(ticket.intersection(draw))
            if hits>=3:
                results[hits]+=1
    return results

# ======================
# File Upload
# ======================
numbers_df=pd.DataFrame(columns=[f"NUMBER DRAWN {i}" for i in range(1,7)])
bonus_series=None
dates=None

uploaded_file=st.file_uploader("📂 Upload a Lotto 6/49 CSV file", type=["csv"])

if uploaded_file:
    raw_df=pd.read_csv(uploaded_file)
    numbers_df, bonus_series, dates=extract_numbers_and_bonus(raw_df)
    if numbers_df is None:
        st.error("❌ Invalid CSV. Must have NUMBER DRAWN 1–6 between 1–49")
        st.stop()
    display_df=numbers_df.copy()
    if bonus_series is not None:
        display_df["BONUS NUMBER"]=bonus_series.astype("Int64").values
    if dates is not None:
        display_df["DATE"]=dates.values
    st.subheader(f"✅ Uploaded Data ({len(numbers_df)} draws):")
    st.dataframe(display_df)

# ======================
# Sidebar Controls
# ======================
st.sidebar.header("⚙️ Global Controls")
max_draws=len(numbers_df)
draw_limit=st.sidebar.slider("Number of past draws to analyze", min_value=1, max_value=max(max_draws,10), value=max(max_draws,10))
numbers_df=numbers_df.tail(draw_limit).reset_index(drop=True)

num_tickets=st.sidebar.slider("Tickets to generate (per tab)",1,12,6)
excluded_str=st.sidebar.text_input("Exclude numbers (comma-separated)","")
excluded={int(x.strip()) for x in excluded_str.split(",") if x.strip().isdigit() and 1<=int(x.strip())<=49}

sum_min,sum_max=st.sidebar.slider("Sum range",60,250,(100,180))
spread_min,spread_max=st.sidebar.slider("Spread range",5,48,(10,40))
odd_mode=st.sidebar.selectbox("Odd/Even constraint", ["Any","Exactly 0 odd","1","2","3","4","5","6"])
odd_count=None if odd_mode=="Any" else int(re.search(r'\d+',odd_mode).group())

# ======================
# Analysis & Ticket Lab
# ======================
counter=compute_frequencies(numbers_df)
hot=[int(n) for n,_ in counter.most_common(6)]
cold=[int(n) for n,_ in counter.most_common()[:-7:-1]]
delta_counter=compute_delta_distribution(numbers_df)
last_draw=set(numbers_df.iloc[-1])
repeats=compute_repeat_frequency(numbers_df)

st.markdown("### 🔥 Hot & ❄️ Cold Numbers")
st.write(f"**Hot:** {hot} | **Cold:** {cold}")

# ======================
# 🎟️ Ticket Lab Tabs
# ======================
st.subheader("🎟️ Generate Lotto Tickets")

tabs = st.tabs(["Balanced", "Delta System", "Zone Coverage", "Repeat Hits", "Random"])

with tabs[0]:
    st.markdown("#### ⚖️ Balanced Tickets (Even/Odd & Sum Spread)")
    tickets = [try_generate_with_constraints(
        generate_balanced_ticket,
        sum_min=sum_min,
        sum_max=sum_max,
        spread_min=spread_min,
        spread_max=spread_max,
        odd_count=odd_count
    ) for _ in range(num_tickets)]
    st.table(pd.DataFrame({"Ticket": tickets}))

with tabs[1]:
    st.markdown("#### 🔺 Delta System Tickets (Based on most frequent gaps)")
    tickets = [try_generate_with_constraints(
        lambda: generate_delta_ticket(delta_counter),
        sum_min=sum_min,
        sum_max=sum_max,
        spread_min=spread_min,
        spread_max=spread_max,
        odd_count=odd_count
    ) for _ in range(num_tickets)]
    st.table(pd.DataFrame({"Ticket": tickets}))

with tabs[2]:
    st.markdown("#### 🗂️ Zone Coverage Tickets (3-zone distribution)")
    tickets = [try_generate_with_constraints(
        generate_zone_ticket,
        sum_min=sum_min,
        sum_max=sum_max,
        spread_min=spread_min,
        spread_max=spread_max,
        odd_count=odd_count
    ) for _ in range(num_tickets)]
    st.table(pd.DataFrame({"Ticket": tickets}))

with tabs[3]:
    st.markdown("#### 🔁 Repeat Hits (Numbers repeating from last draw)")
    tickets = [try_generate_with_constraints(
        lambda: generate_repeat_ticket(last_draw, excluded, repeat_count=2),
        sum_min=sum_min,
        sum_max=sum_max,
        spread_min=spread_min,
        spread_max=spread_max,
        odd_count=odd_count
    ) for _ in range(num_tickets)]
    st.table(pd.DataFrame({"Ticket": tickets}))

with tabs[4]:
    st.markdown("#### 🎲 Random Tickets (Filtered by Exclusions)")
    tickets = [try_generate_with_constraints(
        lambda: generate_ticket(apply_exclusions_to_pool(list(range(1,50)), excluded)),
        sum_min=sum_min,
        sum_max=sum_max,
        spread_min=spread_min,
        spread_max=spread_max,
        odd_count=odd_count
    ) for _ in range(num_tickets)]
    st.table(pd.DataFrame({"Ticket": tickets}))

# ======================
# 📊 Statistics & Plots
# ======================
st.subheader("📊 Number Frequencies")
freq_df = pd.DataFrame(counter.items(), columns=["Number", "Frequency"]).sort_values("Number")
fig = px.bar(freq_df, x="Number", y="Frequency", title="Number Frequency", text="Frequency")
st.plotly_chart(fig, use_container_width=True)

# ======================
# 🔮 Jackpot Simulation
# ======================
st.subheader("🔮 Simulate Strategy Outcomes")
sim_draws = st.slider("Simulation iterations", 100, 5000, 1000, step=100)

if st.button("Run Simulation for Balanced Tickets"):
    results = simulate_strategy(lambda: generate_balanced_ticket(), numbers_df, n=sim_draws)
    st.write("Hits per draw count:", results)

if st.button("Run Simulation for Delta System Tickets"):
    results = simulate_strategy(lambda: generate_delta_ticket(delta_counter), numbers_df, n=sim_draws)
    st.write("Hits per draw count:", results)

# ======================
# 🤖 Optional ML Predictor
# ======================
st.subheader("🤖 ML Prediction (Next Draw Approximation)")
if st.button("Train ML Model"):
    X = numbers_df.shift(1).fillna(0).astype(int)  # Previous draw as features
    y = numbers_df.astype(int)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y)
    last_draw_input = X.iloc[-1].values.reshape(1,-1)
    predicted = model.predict(last_draw_input)[0]
    st.write("Predicted numbers for next draw (approx.):", sorted(predicted.tolist()))
