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
st.write("Analyze historical draws, see hot/cold numbers, generate tickets, and check combinations.")

# --- Helper Functions ---
def extract_numbers_and_bonus(df):
    cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"
    if not all(col in df.columns for col in cols):
        return None, None, None

    main_numbers = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
    bonus_numbers = None
    if bonus_col in df.columns:
        bonus_numbers = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
    # Handle dates
    date_col = next((c for c in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if c in df.columns), None)
    dates = pd.to_datetime(df[date_col], errors='coerce') if date_col else None
    return main_numbers.astype(int), bonus_numbers.astype(int) if bonus_numbers is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df):
    return Counter(numbers_df.values.flatten())

@st.cache_data
def compute_pair_frequencies(numbers_df, limit=500):
    pair_counts = Counter()
    df_subset = numbers_df.tail(limit)
    for _, row in df_subset.iterrows():
        pair_counts.update(combinations(sorted(row.values), 2))
    return pair_counts

@st.cache_data
def compute_triplet_frequencies(numbers_df, limit=500):
    triplet_counts = Counter()
    df_subset = numbers_df.tail(limit)
    for _, row in df_subset.iterrows():
        triplet_counts.update(combinations(sorted(row.values), 3))
    return triplet_counts

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {n: -1 for n in range(1, 50)}
    df = numbers_df.reset_index(drop=True) if dates is None else numbers_df.iloc[dates.argsort()].reset_index(drop=True)
    for idx, row in df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    total = len(df)
    return {n: total - 1 - last_seen[n] if last_seen[n] != -1 else total for n in range(1, 50)}

def most_common_per_draw_position(numbers_df):
    return {col: Counter(numbers_df[col]).most_common(1)[0] for col in numbers_df.columns}

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a Lotto 6/49 CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("CSV must have NUMBER DRAWN 1-6 columns with values 1-49.")
            st.stop()

        # --- Hot & Cold Numbers ---
        counter = compute_frequencies(numbers_df)
        hot = [n for n, _ in counter.most_common(6)]
        cold = [n for n, _ in counter.most_common()[:-7:-1]]

        st.subheader("🔥 Hot Numbers")
        st.write(hot)
        st.subheader("❄️ Cold Numbers")
        st.write(cold)

        # --- Number Gap Analysis (Compact Table) ---
        gaps = compute_number_gaps(numbers_df, dates)
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values(by="Gap", ascending=False)
        st.subheader("🔢 Number Gap Analysis")
        min_gap = st.slider("Show numbers with at least this many draws since last appearance", 0, int(gaps_df["Gap"].max()), int(gaps_df["Gap"].median()))
        st.table(gaps_df[gaps_df["Gap"] >= min_gap].head(10)[["Number", "Gap"]])

        # --- Frequency Charts ---
        st.subheader("📊 Number Frequencies")
        freq_df = pd.DataFrame({"Number": range(1, 50), "Frequency": [counter.get(n, 0) for n in range(1, 50)]})
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        # Pair & Triplet charts
        st.subheader("🔗 Top Number Pairs")
        pair_counts = compute_pair_frequencies(numbers_df)
        pair_df = pd.DataFrame(pair_counts.items(), columns=["Pair","Count"]).sort_values(by="Count", ascending=False).head(20)
        pair_df["Pair"] = pair_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pair_df, y="Pair", x="Count", orientation='h', color="Count", color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        st.subheader("🔗 Top Number Triplets")
        triplet_counts = compute_triplet_frequencies(numbers_df)
        triplet_df = pd.DataFrame(triplet_counts.items(), columns=["Triplet","Count"]).sort_values(by="Count", ascending=False).head(20)
        triplet_df["Triplet"] = triplet_df["Triplet"].apply(lambda x: f"{x[0]} & {x[1]} & {x[2]}")
        fig_triplets = px.bar(triplet_df, y="Triplet", x="Count", orientation='h', color="Count", color_continuous_scale="Cividis")
        fig_triplets.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_triplets, use_container_width=True)

        # --- Ticket Generation ---
        st.subheader("🎟️ Generate Lotto Tickets")
        strategy = st.selectbox("Strategy", ["Pure Random", "Hot Bias", "Cold Bias", "Overdue Bias", "Mixed"])
        num_tickets = st.slider("Number of tickets", 1, 10, 5)

        def generate_ticket(pool):
            return sorted(random.sample(pool, 6))

        generated_tickets = []
        for _ in range(num_tickets):
            if strategy=="Pure Random": pool = list(range(1,50))
            elif strategy=="Hot Bias": pool = hot + random.sample([n for n in range(1,50) if n not in hot], 43)
            elif strategy=="Cold Bias": pool = cold + random.sample([n for n in range(1,50) if n not in cold], 43)
            elif strategy=="Overdue Bias": pool = [n for n, _ in sorted(gaps.items(), key=lambda x:x[1], reverse=True)[:10]] + random.sample([n for n in range(1,50) if n not in gaps],39)
            elif strategy=="Mixed": pool = hot[:3] + cold[:2] + random.sample([n for n in range(1,50) if n not in hot+cold], 49-5)
            ticket = generate_ticket(pool)
            generated_tickets.append(ticket)

        for i, t in enumerate(generated_tickets,1):
            st.write(f"Ticket {i}: {t}")

        # --- Check if a 6-number combination exists ---
        st.subheader("🔍 Check if a Draw Combination Has Already Appeared")
        user_draw = st.text_input("Enter 6 numbers separated by commas (e.g., 5,12,19,23,34,45):", key="check_draw")
        if user_draw.strip():
            try:
                numbers_entered = [int(x.strip()) for x in user_draw.split(",")]
                if len(numbers_entered) != 6: raise ValueError("Enter exactly 6 numbers")
                if not all(1<=n<=49 for n in numbers_entered): raise ValueError("Numbers must be 1-49")
                user_numbers = tuple(sorted(numbers_entered))
                matches_idx = [i for i,row in enumerate(numbers_df.values.tolist()) if tuple(sorted(row))==user_numbers]
                if matches_idx:
                    st.success(f"✅ This combination appeared {len(matches_idx)} time(s)")
                    if dates is not None:
                        st.write("Occurrences:")
                        for i in matches_idx: st.write(f"- {dates.iloc[i].date()}: {list(numbers_df.iloc[i].values)}")
                    else:
                        st.write("Occurrences (row indexes):", [i+1 for i in matches_idx])
                else:
                    st.error("❌ This combination never appeared.")
            except Exception as e:
                st.error(f"⚠️ Invalid input: {e}")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
