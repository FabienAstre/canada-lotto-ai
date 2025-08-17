import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
import plotly.express as px

# --- Streamlit Config ---
st.set_page_config(
    page_title="🎲 Canada Lotto 6/49 Analyzer",
    page_icon="🎲",
    layout="wide"
)
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and check combinations.")

# --- Helper Functions ---

def extract_numbers_and_bonus(df):
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1,7)]
    bonus_col = "BONUS NUMBER"
    
    if not all(col in df.columns for col in main_cols):
        return None, None, None
    
    main_numbers_df = df[main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None
    
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1,49).all():
            bonus_series = None
    
    date_col = next((c for c in ['DATE','Draw Date','Draw_Date','Date'] if c in df.columns), None)
    dates = None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        dates = df[date_col]
    
    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df):
    return Counter(numbers_df.values.flatten())

@st.cache_data
def compute_pair_frequencies(numbers_df, limit=500):
    pair_counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        pair_counts.update(combinations(sorted(row.values), 2))
    return pair_counts

@st.cache_data
def compute_triplet_frequencies(numbers_df, limit=500):
    triplet_counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        triplet_counts.update(combinations(sorted(row.values), 3))
    return triplet_counts

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {n:-1 for n in range(1,50)}
    gaps = {}
    
    numbers_df = numbers_df.reset_index(drop=True) if dates is None else numbers_df.iloc[dates.argsort()].reset_index(drop=True)
    
    for idx, row in numbers_df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    
    total = len(numbers_df)
    for n in range(1,50):
        gaps[n] = total - 1 - last_seen[n] if last_seen[n] != -1 else total
    return gaps

def most_common_per_position(numbers_df):
    result = {}
    for col in numbers_df.columns:
        counts = Counter(numbers_df[col])
        num, freq = counts.most_common(1)[0]
        result[col] = (num, freq)
    return result

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a Lotto 6/49 CSV", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file with draw results.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    df = df.sort_values(by=next((c for c in ['DATE','Draw Date','Draw_Date','Date'] if c in df.columns), df.columns[0]), ascending=True)
    st.subheader(f"Uploaded Data ({len(df)} draws)")
    st.dataframe(df.reset_index(drop=True))
    
    # --- Select Recent Draws ---
    draws_to_use = st.slider("Number of recent draws to analyze", min_value=50, max_value=len(df), value=min(300,len(df)), step=10)
    df_used = df.tail(draws_to_use).reset_index(drop=True)
    
    numbers_df, bonus_series, dates = extract_numbers_and_bonus(df_used)
    if numbers_df is None:
        st.error("CSV must have columns 'NUMBER DRAWN 1' to '6' with values 1-49.")
        st.stop()
    
    # --- Hot & Cold Numbers ---
    st.subheader("🔥 Hot Numbers")
    counter = compute_frequencies(numbers_df)
    hot = [num for num, _ in counter.most_common(6)]
    cols = st.columns(len(hot))
    for i,num in enumerate(hot):
        cols[i].metric("", num)
    
    st.subheader("❄️ Cold Numbers")
    cold = [num for num, _ in counter.most_common()[:-7:-1]]
    cols = st.columns(len(cold))
    for i,num in enumerate(cold):
        cols[i].metric("", num)
    
    # --- Frequency Charts ---
    st.subheader("📊 Number Frequency")
    freq_df = pd.DataFrame({"Number": range(1,50), "Frequency":[counter.get(n,0) for n in range(1,50)]})
    fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", color_continuous_scale="Blues", title=f"Frequency Last {draws_to_use} Draws")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Pair & Triplet Charts ---
    st.subheader("Number Pair Frequency")
    pair_counts = compute_pair_frequencies(numbers_df, limit=draws_to_use)
    pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair","Count"]).sort_values("Count", ascending=False).head(20)
    pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
    fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count", color_continuous_scale="Viridis")
    fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_pairs, use_container_width=True)
    
    st.subheader("Number Triplet Frequency")
    triplet_counts = compute_triplet_frequencies(numbers_df, limit=draws_to_use)
    triplets_df = pd.DataFrame(triplet_counts.items(), columns=["Triplet","Count"]).sort_values("Count", ascending=False).head(20)
    triplets_df["Triplet"] = triplets_df["Triplet"].apply(lambda x: f"{x[0]} & {x[1]} & {x[2]}")
    fig_triplets = px.bar(triplets_df, y="Triplet", x="Count", orientation='h', color="Count", color_continuous_scale="Cividis")
    fig_triplets.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_triplets, use_container_width=True)
    
    # --- Number Gap Analysis (Table Design) ---
    st.subheader("🔢 Number Gap Analysis")
    gaps = compute_number_gaps(numbers_df, dates)
    gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Draws Since Last": list(gaps.values())}).sort_values("Draws Since Last", ascending=False)
    
    threshold = st.slider("Show numbers with at least this many draws since last appearance", 0, max(gaps.values()), value=20)
    st.dataframe(gaps_df[gaps_df["Draws Since Last"] >= threshold].reset_index(drop=True))
    
    # --- Ticket Generator ---
    st.subheader("🎟️ Generate Lotto Tickets")
    strategy = st.selectbox("Strategy", ["Pure Random","Bias: Hot","Bias: Cold","Bias: Overdue","Mixed"])
    num_tickets = st.slider("Number of tickets", 1, 10, 5)
    
    def generate_ticket(pool):
        return sorted(random.sample(pool,6))
    
    generated_tickets = []
    for _ in range(num_tickets):
        if strategy=="Pure Random":
            pool = list(range(1,50))
        elif strategy=="Bias: Hot":
            pool = hot + random.sample([n for n in range(1,50) if n not in hot], 43)
        elif strategy=="Bias: Cold":
            pool = cold + random.sample([n for n in range(1,50) if n not in cold], 43)
        elif strategy=="Bias: Overdue":
            top_gap = [n for n,_ in sorted(gaps.items(), key=lambda x:x[1], reverse=True)[:10]]
            pool = top_gap + random.sample([n for n in range(1,50) if n not in top_gap], 39)
        elif strategy=="Mixed":
            pool = hot[:3] + cold[:2] + random.sample([n for n in range(1,50) if n not in hot[:3]+cold[:2]], 49-5)
        generated_tickets.append(generate_ticket(pool))
    
    for idx,t in enumerate(generated_tickets,1):
        st.write(f"Ticket {idx}: {t}")
    
    # --- Check if a Combination Already Appeared ---
    st.subheader("🔍 Check if a Draw Combination Has Already Appeared (All History)")
    user_draw = st.text_input("Enter 6 numbers separated by commas (e.g., 5,12,19,23,34,45)", key="combo_check")
    
    if user_draw.strip():
        try:
            numbers_entered = [int(x.strip()) for x in user_draw.split(",")]
            if len(numbers_entered)!=6 or not all(1<=n<=49 for n in numbers_entered):
                raise ValueError("Enter exactly 6 numbers between 1-49.")
            user_tuple = tuple(sorted(numbers_entered))
            
            matches_idx = [i for i,row in enumerate(numbers_df.values.tolist()) if tuple(sorted(row))==user_tuple]
            if matches_idx:
                st.success(f"✅ This combination appeared {len(matches_idx)} time(s)!")
                st.write("Occurrences (row indexes):", [i+1 for i in matches_idx])
            else:
                st.error("❌ This combination never appeared.")
        except Exception as e:
            st.error(f"⚠️ {e}")
    
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
