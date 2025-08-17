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
    """Extract main numbers and bonus column, optionally parse dates"""
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"
    
    if not all(col in df.columns for col in main_cols):
        return None, None, None

    numbers_df = df[main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None
    
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    date_col = next((col for col in ['DATE','Draw Date','Draw_Date','Date'] if col in df.columns), None)
    dates = None
    if date_col:
        import re
        df[date_col] = df[date_col].astype(str).apply(lambda x: re.sub(r'(\d+)(st|nd|rd|th)', r'\1', x))
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        dates = df[date_col]

    return numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df):
    all_numbers = numbers_df.values.flatten()
    return Counter(all_numbers)

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
    df = numbers_df.copy()
    if dates is not None:
        df = df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    for idx, row in df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    total_draws = len(df)
    gaps = {n: total_draws - 1 - last_seen[n] if last_seen[n] != -1 else total_draws for n in range(1,50)}
    return gaps

def most_common_per_position(numbers_df):
    result = {}
    for col in numbers_df.columns:
        num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (num, freq)
    return result

def generate_ticket(pool):
    return sorted(random.sample(pool, 6))

def generate_ml_ticket(must_include, predicted_numbers):
    ticket = must_include.copy()
    pool = [n for n in predicted_numbers if n not in ticket]
    needed = 6 - len(ticket)
    ticket += random.sample(pool, needed) if len(pool) >= needed else pool + random.sample([n for n in range(1,50) if n not in ticket], 6-len(ticket))
    swap_count = random.randint(1,2)
    for _ in range(swap_count):
        idx = random.randint(0,5)
        if ticket[idx] in must_include: continue
        available = [n for n in range(1,50) if n not in ticket]
        if not available: break
        ticket[idx] = random.choice(available)
    return sorted(ticket)

def generate_position_based_ticket(position_dict, must_include=[]):
    ticket = list({v[0] for v in position_dict.values()})
    for n in must_include:
        if n not in ticket: ticket.append(n)
    while len(ticket) < 6:
        candidate = random.randint(1,49)
        if candidate not in ticket: ticket.append(candidate)
    while len(ticket) > 6:
        removable = [n for n in ticket if n not in must_include]
        if not removable: break
        ticket.remove(random.choice(removable))
    swap_count = random.randint(1,2)
    for _ in range(swap_count):
        idx = random.randint(0,5)
        if ticket[idx] in must_include: continue
        available = [n for n in range(1,50) if n not in ticket]
        if not available: break
        ticket[idx] = random.choice(available)
    return sorted(ticket)

# -------------------
# File Upload
# -------------------
uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV must include columns NUMBER DRAWN 1-6, optional BONUS NUMBER, optional DATE"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("Invalid CSV. Make sure columns NUMBER DRAWN 1-6 exist with numbers 1-49.")
            st.stop()

        st.subheader(f"Uploaded Data ({len(df)} draws):")
        st.dataframe(df.reset_index(drop=True))

        # --- Hot & Cold Numbers ---
        st.subheader("🔥 Hot & ❄️ Cold Numbers")
       counter = compute_frequencies(numbers_df)
hot = [int(n) for n,_ in counter.most_common(6)]
cold = [int(n) for n,_ in counter.most_common()[:-7:-1]]
st.write(f"Hot Numbers: {hot}")
st.write(f"Cold Numbers: {cold}")

        # --- Most Common per Position ---
        st.subheader("Most Common Numbers by Draw Position")
        pos_common = most_common_per_position(numbers_df)
        for pos, (num,freq) in pos_common.items():
            st.write(f"{pos}: {num} (appeared {freq} times)")

        # --- Frequency Chart ---
        freq_df = pd.DataFrame({"Number": list(range(1,50)), "Frequency": [counter.get(n,0) for n in range(1,50)]})
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", color_continuous_scale="Blues", title="Number Frequency")
        st.plotly_chart(fig, use_container_width=True)

        # --- Pair & Triplet Charts ---
        st.subheader("Pair Frequency (top 20)")
        pair_counts = compute_pair_frequencies(numbers_df)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair","Count"]).sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, x="Count", y="Pair", orientation="h", color="Count", color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        st.subheader("Triplet Frequency (top 20)")
        triplet_counts = compute_triplet_frequencies(numbers_df)
        triplets_df = pd.DataFrame(triplet_counts.items(), columns=["Triplet","Count"]).sort_values(by="Count", ascending=False).head(20)
        triplets_df["Triplet"] = triplets_df["Triplet"].apply(lambda x: f"{x[0]} & {x[1]} & {x[2]}")
        fig_triplets = px.bar(triplets_df, x="Count", y="Triplet", orientation="h", color="Count", color_continuous_scale="Cividis")
        fig_triplets.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_triplets, use_container_width=True)

        # --- Number Gap Analysis ---
        st.subheader("🔢 Number Gap Analysis")
        gaps = compute_number_gaps(numbers_df, dates)
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values(by="Gap", ascending=False)
        # Show only top 10
        top_gaps_df = gaps_df.head(10)
        st.table(top_gaps_df)
        fig_gap = px.bar(top_gaps_df, x="Number", y="Gap", color="Gap", color_continuous_scale="Oranges", text="Gap", title="Top 10 Number Gaps")
        fig_gap.update_traces(textposition='outside')
        st.plotly_chart(fig_gap, use_container_width=True)

        overdue_threshold = st.slider("Highlight numbers very overdue:", min_value=0, max_value=int(gaps_df["Gap"].max()), value=int(gaps_df["Gap"].quantile(0.75)))
        overdue_numbers = gaps_df[gaps_df["Gap"] >= overdue_threshold]["Number"].tolist()
        st.info(f"⚠️ Very overdue numbers: {overdue_numbers}")

        # --- Ticket Generator ---
        st.subheader("🎟️ Generate Tickets")
        strategy = st.selectbox("Ticket Generation Strategy", ["Pure Random","Hot Bias","Cold Bias","Overdue Bias","Mixed"])
        num_tickets = st.slider("Number of tickets to generate", 1, 10, 5)
        tickets = []
        for _ in range(num_tickets):
            if strategy=="Pure Random": pool=list(range(1,50))
            elif strategy=="Hot Bias": pool=hot + [n for n in range(1,50) if n not in hot]
            elif strategy=="Cold Bias": pool=cold + [n for n in range(1,50) if n not in cold]
            elif strategy=="Overdue Bias": pool = sorted(gaps.items(), key=lambda x:x[1],reverse=True)[:10]; pool=[n for n,_ in pool]+[n for n in range(1,50) if n not in [n for n,_ in pool]]
            elif strategy=="Mixed": pool = hot[:3]+cold[:2]+[n for n in range(1,50) if n not in hot[:3]+cold[:2]]
            tickets.append(generate_ticket(pool))
        for idx, t in enumerate(tickets,1): st.write(f"Ticket {idx}: {t}")

        # --- ML-based Prediction ---
        st.subheader("🧠 ML-based Prediction (Experimental)")
        must_include = st.multiselect("Numbers to include in every ML ticket", options=list(range(1,50)), default=[])
        num_ml = st.slider("ML tickets to generate", 1, 10, 3)
        predicted_numbers = [n for n,_ in counter.most_common(12)]
        for i in range(num_ml):
            ml_ticket = generate_ml_ticket(must_include, predicted_numbers)
            st.write(f"ML Ticket {i+1}: {ml_ticket}")

        # --- Position-based Prediction ---
        st.subheader("🎯 Position-based Prediction")
        must_include_pos = st.multiselect("Numbers to always include", options=list(range(1,50)), default=[])
        num_pos = st.slider("Position-based tickets to generate", 1, 10, 3)
        for i in range(num_pos):
            pos_ticket = generate_position_based_ticket(pos_common, must_include_pos)
            st.write(f"Position-based Ticket {i+1}: {pos_ticket}")

        # --- Check if Draw Combination Exists ---
        st.subheader("🔍 Check if Draw Combination Already Appeared")
        user_draw = st.text_input("Enter 6 numbers separated by commas (e.g., 5,12,19,23,34,45):", key="check_draw")
        if user_draw.strip():
            try:
                numbers_entered = tuple(sorted(int(x.strip()) for x in user_draw.split(",")))
                if len(numbers_entered) != 6: raise ValueError("Enter exactly 6 numbers.")
                past_draws = [tuple(sorted(row)) for row in numbers_df.values.tolist()]
                occurrences = past_draws.count(numbers_entered)
                if occurrences>0: st.success(f"✅ This combination appeared {occurrences} time(s) in history!")
                else: st.error("❌ This combination never appeared.")
            except Exception as e:
                st.error(f"⚠️ Invalid input: {e}")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
