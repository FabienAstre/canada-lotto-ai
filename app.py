import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
import random
import plotly.express as px

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(
    page_title="🎲 Canada Lotto 6/49 Analyzer",
    page_icon="🎲",
    layout="wide"
)
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see predictions.")

# -------------------
# Helper Functions
# -------------------
def extract_numbers_and_bonus(df):
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"
    if not all(col in df.columns for col in main_cols):
        return None, None, None
    numbers_df = df[main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
    date_col = next((col for col in ['DATE','Draw Date','Draw_Date','Date'] if col in df.columns), None)
    dates = None
    if date_col:
        import re
        df[date_col] = df[date_col].astype(str).apply(lambda x: re.sub(r'(\d+)(st|nd|rd|th)', r'\1', x))
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        dates = df[date_col]
    return numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df): return Counter(numbers_df.values.flatten())

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

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {n: -1 for n in range(1,50)}
    df = numbers_df.copy().reset_index(drop=True)
    if dates is not None: df = df.iloc[dates.argsort()].reset_index(drop=True)
    for idx, row in df.iterrows():
        for n in row.values: last_seen[n] = idx
    total_draws = len(df)
    return {n: total_draws - 1 - last_seen[n] if last_seen[n] != -1 else total_draws for n in range(1,50)}

def most_common_per_position(numbers_df):
    result = {}
    for col in numbers_df.columns:
        num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (int(num), int(freq))
    return result

# --- Ticket Generators ---
def generate_ticket(pool): return sorted(random.sample(pool, 6))

def generate_balanced_ticket():
    while True:
        ticket = random.sample(range(1,17),2)+random.sample(range(17,34),2)+random.sample(range(34,50),2)
        ticket = sorted(ticket)
        odds = sum(1 for n in ticket if n%2==1); evens = 6-odds
        total = sum(ticket)
        if odds==3 and evens==3 and 100<=total<=180: return ticket

# --- NEW: Delta System ---
def compute_delta_distribution(numbers_df):
    deltas=[]
    for row in numbers_df.values:
        row=sorted(row)
        deltas+=[row[i+1]-row[i] for i in range(5)]
    return Counter(deltas)

def generate_delta_ticket(delta_counter):
    deltas=[d for d,_ in delta_counter.most_common(10)]
    while True:
        start=random.randint(1,20)
        ticket=[start]
        for _ in range(5): ticket.append(ticket[-1]+random.choice(deltas))
        ticket=[n for n in ticket if 1<=n<=49]
        if len(ticket)==6: return sorted(ticket)

# --- NEW: Zone Coverage ---
def generate_zone_ticket(mode="3-zone"):
    if mode=="3-zone":
        return sorted(random.sample(range(1,17),2)+random.sample(range(17,34),2)+random.sample(range(34,50),2))
    else:
        return sorted(random.sample(range(1,13),1)+random.sample(range(13,25),2)+random.sample(range(25,37),2)+random.sample(range(37,50),1))

# --- NEW: Constraints ---
def passes_constraints(ticket,sum_min,sum_max,spread_min,spread_max,odd_count):
    total=sum(ticket); spread=max(ticket)-min(ticket)
    odds=sum(1 for n in ticket if n%2==1); evens=6-odds
    return (sum_min<=total<=sum_max and spread_min<=spread<=spread_max and odds==odd_count and evens==6-odd_count)

# --- NEW: Smart Exclusion ---
def exclude_numbers(ticket,excluded): return [n for n in ticket if n not in excluded]

# --- NEW: Repeat Hit ---
def compute_repeat_frequency(numbers_df):
    past=[set(r) for r in numbers_df.values.tolist()]
    repeats=Counter()
    for i in range(1,len(past)):
        for n in past[i].intersection(past[i-1]): repeats[n]+=1
    return repeats

def generate_repeat_ticket(last_draw,repeats,repeat_count=1):
    repeat_nums=random.sample(list(last_draw),repeat_count)
    pool=[n for n in range(1,50) if n not in repeat_nums]
    return sorted(repeat_nums+random.sample(pool,6-repeat_count))

# --- NEW: Simulation ---
def simulate_strategy(strategy_func,numbers_df,n=500):
    past=[set(r) for r in numbers_df.values.tolist()]
    results={3:0,4:0,5:0,6:0}
    for _ in range(n):
        ticket=set(strategy_func())
        for draw in past:
            hits=len(ticket.intersection(draw))
            if hits>=3: results[hits]+=1
    return results

# -------------------
# File Upload
# -------------------
uploaded_file=st.file_uploader("Upload Lotto 6/49 CSV",type=["csv"])
if uploaded_file:
    try:
        df=pd.read_csv(uploaded_file)
        numbers_df,bonus_series,dates=extract_numbers_and_bonus(df)
        if numbers_df is None: st.error("Invalid CSV."); st.stop()

        # --- NEW: slider to limit draws ---
        max_draws=len(numbers_df)
        draw_limit=st.slider("Number of past draws to analyze",10,max_draws,max_draws)
        numbers_df=numbers_df.tail(draw_limit)

        st.subheader(f"Uploaded Data ({len(numbers_df)} draws analyzed):")
        st.dataframe(numbers_df.reset_index(drop=True))

        # Hot/Cold
        st.subheader("🔥 Hot & ❄️ Cold Numbers")
        counter=compute_frequencies(numbers_df)
        hot=[n for n,_ in counter.most_common(6)]
        cold=[n for n,_ in counter.most_common()[:-7:-1]]
        st.write(f"Hot: {hot}")
        st.write(f"Cold: {cold}")

        # Ticket Generator
        st.subheader("🎟️ Generate Tickets")
        num_tickets=st.slider("Tickets to generate",1,10,5)
        strategy=st.selectbox("Strategy",[
            "Pure Random","Balanced","Delta System","Zone Coverage (3-zone)","Zone Coverage (Quartiles)","Repeat Hit"
        ])

        excluded_str=st.text_input("Exclude numbers (comma separated):","")
        excluded=[int(x) for x in excluded_str.split(",") if x.strip().isdigit()]
        sum_min,sum_max=st.slider("Sum range",60,250,(100,180))
        spread_min,spread_max=st.slider("Spread range",5,48,(10,40))
        odd_count=st.slider("Odd numbers required",0,6,3)

        tickets=[]
        delta_counter=compute_delta_distribution(numbers_df)
        last_draw=set(numbers_df.iloc[-1])
        repeats=compute_repeat_frequency(numbers_df)

        for _ in range(num_tickets):
            if strategy=="Pure Random": ticket=generate_ticket(list(range(1,50)))
            elif strategy=="Balanced": ticket=generate_balanced_ticket()
            elif strategy=="Delta System": ticket=generate_delta_ticket(delta_counter)
            elif strategy=="Zone Coverage (3-zone)": ticket=generate_zone_ticket("3-zone")
            elif strategy=="Zone Coverage (Quartiles)": ticket=generate_zone_ticket("quartiles")
            elif strategy=="Repeat Hit": ticket=generate_repeat_ticket(last_draw,repeats,2)
            else: ticket=generate_ticket(list(range(1,50)))
            ticket=exclude_numbers(ticket,excluded)
            if passes_constraints(ticket,sum_min,sum_max,spread_min,spread_max,odd_count):
                tickets.append(ticket)

        st.subheader("🎯 Generated Tickets")
        for i,t in enumerate(tickets,1): st.write(f"Ticket {i}: {t}")

        # Simulation
        st.subheader("📊 Jackpot Pattern Simulation")
        sim_strategy=st.selectbox("Simulate strategy",["Delta System","Balanced","Pure Random"])
        if st.button("Run Simulation"):
            if sim_strategy=="Delta System": func=lambda: generate_delta_ticket(delta_counter)
            elif sim_strategy=="Balanced": func=generate_balanced_ticket
            else: func=lambda: generate_ticket(list(range(1,50)))
            results=simulate_strategy(func,numbers_df,500)
            st.json(results)

    except Exception as e: st.error(f"❌ Error: {e}")
else:
    st.info("Upload a CSV file to begin.")
