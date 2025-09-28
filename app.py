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
st.write("Analyze historical draws, identify patterns, generate tickets, and backtest strategies.")

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

    # Bonus number
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors="coerce")
        bonus_series = bonus_series.where(bonus_series.between(1,49))

    # Flexible date parsing
    date_col = next((col for col in ["DATE","Draw Date","Draw_Date","Date"] if col in df.columns), None)
    dates = None
    if date_col:
        import re
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
    return {n: (total_draws-1-last_seen[n]) if last_seen[n] != -1 else total_draws for n in range(1,50)}

def most_common_per_position(numbers_df: pd.DataFrame):
    result = {}
    for col in numbers_df.columns:
        num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (int(num), int(freq))
    return result

# -----------------
# Ticket Generators
# -----------------

def generate_ticket(pool: list[int]):
    pool = [int(n) for n in pool]
    if len(pool) < 6:
        pool = [n for n in range(1,50)]
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

def generate_zone_ticket(mode="3-zone"):
    if mode=="3-zone":
        low=random.sample(range(1,17),2)
        mid=random.sample(range(17,34),2)
        high=random.sample(range(34,50),2)
        return sorted(low+mid+high)
    q1=random.sample(range(1,13),1)
    q2=random.sample(range(13,25),2)
    q3=random.sample(range(25,37),2)
    q4=random.sample(range(37,50),1)
    return sorted(q1+q2+q3+q4)

def passes_constraints(ticket, sum_min,sum_max,spread_min,spread_max,odd_count=None):
    total=sum(ticket)
    spread=max(ticket)-min(ticket)
    odds=sum(1 for n in ticket if n%2==1)
    if odd_count is not None and odds!=odd_count:
        return False
    if not (sum_min<=total<=sum_max):
        return False
    if not (spread_min<=spread<=spread_max):
        return False
    return True

def apply_exclusions_to_pool(pool, excluded:set):
    pool=[n for n in pool if n not in excluded]
    if len(pool)<6:
        pool=[n for n in range(1,50) if n not in excluded]
    return pool

@st.cache_data
def compute_repeat_frequency(numbers_df: pd.DataFrame):
    past=[set(row) for row in numbers_df.values.tolist()]
    repeats=Counter()
    for i in range(1,len(past)):
        for n in past[i].intersection(past[i-1]):
            repeats[n]+=1
    return repeats

def generate_repeat_ticket(last_draw:set, excluded:set, repeat_count:int=1):
    candidates=list(last_draw-excluded)
    if len(candidates)<repeat_count:
        repeat_count=max(0,len(candidates))
    chosen_repeats=random.sample(candidates,repeat_count) if repeat_count>0 else []
    pool=[n for n in range(1,50) if n not in set(chosen_repeats)|excluded]
    rest=random.sample(pool,6-len(chosen_repeats))
    return sorted(chosen_repeats+rest)

def simulate_strategy(strategy_func, numbers_df: pd.DataFrame, n=1000):
    past_draws=[set(row) for row in numbers_df.values.tolist()]
    results={3:0,4:0,5:0,6:0}
    for _ in range(n):
        ticket=set(strategy_func())
        for draw in past_draws:
            hits=len(ticket.intersection(draw))
            if hits>=3:
                results[hits]+=1
    return results

def try_generate_with_constraints(gen_callable, *, sum_min,sum_max,spread_min,spread_max,odd_count,max_tries=200):
    last_ticket=None
    for _ in range(max_tries):
        t=gen_callable()
        last_ticket=t
        if passes_constraints(t,sum_min,sum_max,spread_min,spread_max,odd_count):
            return t
    return last_ticket

# ======================
# CSV Upload
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
    raw_df=pd.read_csv(uploaded_file)
    numbers_df, bonus_series, dates = extract_numbers_and_bonus(raw_df)
    if numbers_df is None:
        st.error("❌ Invalid CSV. Ensure columns NUMBER DRAWN 1–6 exist with values between 1 and 49.")
        st.stop()
    display_df = numbers_df.reset_index(drop=True)
    if bonus_series is not None and len(bonus_series)==len(display_df):
        display_df["BONUS NUMBER"]=bonus_series.reset_index(drop=True).astype("Int64")
    if dates is not None and len(dates)==len(display_df):
        display_df["DATE"]=dates.reset_index(drop=True).astype(str)
    st.subheader(f"✅ Uploaded Data ({len(raw_df)} draws):")
    st.dataframe(display_df)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

# -----------------
# Sidebar controls
# -----------------
st.sidebar.header("⚙️ Global Controls")
max_draws=len(numbers_df)
draw_limit=st.sidebar.slider("Use last N draws", min_value=10, max_value=max_draws, value=min(200,max_draws))
tickets_to_generate=st.sidebar.number_input("Tickets to generate per strategy", min_value=1, max_value=50, value=5)

# ======================
# Tabs
# ======================
tabs = st.tabs([
    "🔥 Hot / ❄️ Cold / ⏳ Overdue",
    "Δ Delta System",
    "🏙️ Zone Coverage",
    "📊 Sum & Spread",
    "🚫 Smart Exclusion",
    "🔁 Repeat Hits",
    "💰 Jackpot Simulation",
    "🤖 ML Predictions",
])

# -----------------
# Tab 1: Hot / Cold / Overdue
# -----------------
with tabs[0]:
    st.subheader("🔥 Hot Numbers / ❄️ Cold Numbers / ⏳ Overdue Numbers")
    freq = compute_frequencies(numbers_df.tail(draw_limit))
    most_common = freq.most_common(10)
    least_common = freq.most_common()[:-11:-1]
    gaps = compute_number_gaps(numbers_df.tail(draw_limit))
    st.write("Top 10 Hot Numbers:", [n for n,_ in most_common])
    st.write("Top 10 Cold Numbers:", [n for n,_ in least_common])
    st.write("Top 10 Overdue Numbers:", sorted(gaps, key=lambda k: gaps[k], reverse=True)[:10])

# -----------------
# Tab 2: Δ Delta
# -----------------
with tabs[1]:
    st.subheader("Δ Delta Distribution")
    delta_counter=compute_delta_distribution(numbers_df.tail(draw_limit))
    df_delta=pd.DataFrame(delta_counter.items(), columns=["Delta","Count"]).sort_values("Delta")
    fig=px.bar(df_delta, x="Delta", y="Count", title="Delta Frequency (difference between consecutive numbers)")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Generate Delta-Based Tickets:")
    delta_tickets=[generate_delta_ticket(delta_counter) for _ in range(tickets_to_generate)]
    st.dataframe(pd.DataFrame(delta_tickets, columns=[f"Number {i}" for i in range(1,7)]))

# -----------------
# Tab 3: Zone Coverage
# -----------------
with tabs[2]:
    st.subheader("🏙️ Zone / Cluster Tickets")
    zone_tickets=[generate_zone_ticket() for _ in range(tickets_to_generate)]
    st.dataframe(pd.DataFrame(zone_tickets, columns=[f"Number {i}" for i in range(1,7)]))

# -----------------
# Tab 4: Sum & Spread
# -----------------
with tabs[3]:
    st.subheader("📊 Sum & Spread Filtered Tickets")
    sum_min = st.number_input("Sum Minimum", min_value=21, max_value=294, value=100)
    sum_max = st.number_input("Sum Maximum", min_value=21, max_value=294, value=180)
    spread_min = st.number_input("Spread Minimum", min_value=0, max_value=48, value=10)
    spread_max = st.number_input("Spread Maximum", min_value=0, max_value=48, value=35)
    odd_count = st.number_input("Number of odd numbers (leave -1 to ignore)", min_value=-1, max_value=6, value=-1)
    tickets=[]
    for _ in range(tickets_to_generate):
        t = try_generate_with_constraints(generate_balanced_ticket,
                                         sum_min=sum_min,
                                         sum_max=sum_max,
                                         spread_min=spread_min,
                                         spread_max=spread_max,
                                         odd_count=(odd_count if odd_count>=0 else None))
        tickets.append(t)
    st.dataframe(pd.DataFrame(tickets, columns=[f"Number {i}" for i in range(1,7)]))

# -----------------
# Tab 5: Smart Exclusion
# -----------------
with tabs[4]:
    st.subheader("🚫 Exclude Numbers from Ticket Generation")
    excluded_nums = st.multiselect("Select numbers to exclude", options=list(range(1,50)))
    smart_tickets=[]
    for _ in range(tickets_to_generate):
        pool = apply_exclusions_to_pool(list(range(1,50)), set(excluded_nums))
        t = generate_ticket(pool)
        smart_tickets.append(t)
    st.dataframe(pd.DataFrame(smart_tickets, columns=[f"Number {i}" for i in range(1,7)]))

# -----------------
# Tab 6: Repeat Hits
# -----------------
with tabs[5]:
    st.subheader("🔁 Generate Tickets with Repeat Numbers from Last Draw")
    last_draw = set(numbers_df.tail(1).values.flatten())
    repeat_count = st.slider("Number of repeat numbers to include", min_value=0, max_value=6, value=2)
    repeat_tickets=[]
    for _ in range(tickets_to_generate):
        t=generate_repeat_ticket(last_draw, excluded=set(excluded_nums), repeat_count=repeat_count)
        repeat_tickets.append(t)
    st.dataframe(pd.DataFrame(repeat_tickets, columns=[f"Number {i}" for i in range(1,7)]))

# -----------------
# Tab 7: Jackpot Simulation
# -----------------
with tabs[6]:
    st.subheader("💰 Backtest Strategy against Past Draws")
    results=simulate_strategy(lambda: generate_ticket(list(range(1,50))), numbers_df.tail(draw_limit), n=500)
    st.write("Hits Distribution over past draws (3–6 matches):", results)

# -----------------
# Tab 8: ML Predictions
# -----------------
with tabs[7]:
    st.subheader("🤖 ML-Based Predictions")
    last_features = numbers_df.tail(50)  # last 50 draws
    X = []
    y = []
    for i in range(len(last_features)-1):
        X.append(last_features.iloc[i].values)
        y.append(last_features.iloc[i+1].values)
    X = np.array(X)
    y = np.array(y)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42))
    model.fit(X,y)
    ml_tickets=[]
    for _ in range(tickets_to_generate):
        base = numbers_df.tail(1).values.flatten().reshape(1,-1)
        pred = model.predict(base)[0]
        # Random perturbation
        pred = [min(max(int(n+random.choice([-1,0,1])),1),49) for n in pred]
        while len(set(pred))<6:
            pred[random.randint(0,5)] = random.randint(1,49)
        ml_tickets.append(sorted(pred))
    st.dataframe(pd.DataFrame(ml_tickets, columns=[f"Number {i}" for i in range(1,7)]))

# -----------------
# Most Common per Position
# -----------------
st.subheader("📌 Most Common Numbers per Position")
pos_common = most_common_per_position(numbers_df.tail(draw_limit))
pos_df = pd.DataFrame(pos_common.values(), columns=["Number","Count"], index=pos_common.keys())
st.dataframe(pos_df)

# -----------------
# CSV Export
# -----------------
st.subheader("📥 Download Generated Tickets")
all_tickets = delta_tickets + zone_tickets + tickets + smart_tickets + repeat_tickets + ml_tickets
if all_tickets:
    export_df = pd.DataFrame(all_tickets, columns=[f"Number {i}" for i in range(1,7)])
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Tickets CSV", csv_data, file_name="generated_tickets.csv", mime="text/csv")
