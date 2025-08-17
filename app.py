import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
import plotly.express as px

# Streamlit config
st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see predictions.")

# --- Helper Functions ---
def extract_numbers_and_bonus(df):
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1,7)]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in main_cols):
        return None, None, None

    numbers_df = df[main_cols].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna().astype(int)
    else:
        bonus_series = None

    date_col = next((c for c in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if c in df.columns), None)
    dates = pd.to_datetime(df[date_col], errors='coerce') if date_col else None

    return numbers_df, bonus_series, dates

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
    last_seen = {n: -1 for n in range(1,50)}
    df = numbers_df.copy()
    if dates is not None:
        df = df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    total = len(df)
    return {n: total-1-last_seen[n] if last_seen[n]!=-1 else total for n in range(1,50)}

def most_common_per_draw_position(numbers_df):
    result = {}
    for col in numbers_df.columns:
        counts = Counter(numbers_df[col])
        most_common_num, freq = counts.most_common(1)[0]
        result[col] = (most_common_num, freq)
    return result

# --- Main App ---
uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to 6, BONUS NUMBER, and optional DATE"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        date_col = next((c for c in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(date_col)

        st.subheader(f"Uploaded Data ({len(df)} draws)")
        st.dataframe(df.reset_index(drop=True))

        # Select how many recent draws to analyze
        max_draws = len(df)
        draws_to_use = st.slider("Number of recent draws to analyze", min_value=50, max_value=max_draws, value=min(300,max_draws), step=10)
        df_used = df.tail(draws_to_use).reset_index(drop=True)

        # Extract numbers
        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df_used)
        if numbers_df is None:
            st.error("CSV must have valid columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6'")
            st.stop()

        # Frequencies & stats
        counter = compute_frequencies(numbers_df)
        hot_counts = counter.most_common(6)
        cold_counts = counter.most_common()[:-7:-1]
        gaps = compute_number_gaps(numbers_df, dates)
        position_most_common = most_common_per_draw_position(numbers_df)

        # --- Hot & Cold Numbers ---
        st.subheader("🔥 Hot Numbers")
        cols = st.columns(len(hot_counts))
        for i, (num, freq) in enumerate(hot_counts):
            cols[i].metric(f"#{num}", f"{freq} times")

        st.subheader("❄️ Cold Numbers")
        cols = st.columns(len(cold_counts))
        for i, (num, freq) in enumerate(cold_counts):
            cols[i].metric(f"#{num}", f"{freq} times")

        # --- Number Gap Analysis ---
        st.subheader("🔢 Number Gap Analysis")
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values("Gap", ascending=False)

        min_gap = st.slider(
            "Show numbers with at least this many draws since last appearance",
            0, int(max(gaps.values())), 20
        )
        filtered_gaps = gaps_df[gaps_df["Gap"] >= min_gap].reset_index(drop=True)
        st.dataframe(filtered_gaps)

        # Visualize gap chart
        fig_gap = px.bar(
            filtered_gaps,
            x="Number",
            y="Gap",
            color="Gap",
            color_continuous_scale="Oranges",
            text="Gap",
            title=f"Numbers with Gap ≥ {min_gap} Draws"
        )
        fig_gap.update_traces(textposition='outside')
        st.plotly_chart(fig_gap, use_container_width=True)

        # Highlight very overdue numbers
        overdue_threshold = st.slider(
            "Highlight numbers considered 'very overdue'",
            0, int(gaps_df["Gap"].max()), int(gaps_df["Gap"].quantile(0.75))
        )
        overdue_numbers = filtered_gaps[filtered_gaps["Gap"] >= overdue_threshold]["Number"].tolist()
        if overdue_numbers:
            st.info(f"⚠️ Numbers very overdue: {overdue_numbers}")

        # --- Ticket Generator ---
        st.subheader("🎟️ Generate Lotto Tickets")
        strategy = st.selectbox("Strategy", ["Pure Random","Bias: Hot","Bias: Cold","Bias: Overdue","Mixed"])
        num_tickets = st.slider("Number of tickets",1,10,5)

        def generate_ticket(pool):
            return sorted(random.sample(pool, 6))

        generated_tickets = []
        for _ in range(num_tickets):
            if strategy=="Pure Random":
                pool = list(range(1,50))
            elif strategy=="Bias: Hot":
                pool = [n for n,_ in hot_counts] + [n for n in range(1,50) if n not in [x for x,_ in hot_counts]]
            elif strategy=="Bias: Cold":
                pool = [n for n,_ in cold_counts] + [n for n in range(1,50) if n not in [x for x,_ in cold_counts]]
            elif strategy=="Bias: Overdue":
                top_gap = [n for n,g in sorted(gaps.items(), key=lambda x:x[1], reverse=True)[:10]]
                pool = top_gap + [n for n in range(1,50) if n not in top_gap]
            elif strategy=="Mixed":
                pool = [n for n,_ in hot_counts[:3]] + [n for n,_ in cold_counts[:2]]
                pool += [n for n in range(1,50) if n not in pool]
            generated_tickets.append(generate_ticket(pool))

        st.write("🎰 Generated Tickets:")
        for i,ticket in enumerate(generated_tickets,1):
            st.write(f"Ticket {i}: {ticket}")

        # --- ML-Based Prediction ---
        st.subheader("🧠 ML-Based Prediction (Experimental)")
        must_include = st.multiselect("Numbers to include in ML tickets", options=list(range(1,50)), default=[])
        num_ml_tickets = st.slider("ML predicted tickets",1,10,3)
        predicted_numbers = [num for num,_ in counter.most_common(12)]
        st.write("Base predicted numbers:", predicted_numbers)

        def generate_ml_ticket(must_include,predicted):
            ticket = must_include.copy()
            pool = [n for n in predicted if n not in ticket]
            while len(ticket)<6 and pool:
                ticket.append(pool.pop(0))
            while len(ticket)<6:
                candidate = random.randint(1,49)
                if candidate not in ticket:
                    ticket.append(candidate)
            return sorted(ticket)

        for i in range(num_ml_tickets):
            st.write(f"ML Ticket {i+1}: {generate_ml_ticket(must_include, predicted_numbers)}")

        # --- Position-Based Prediction ---
        st.subheader("🎯 Position-Based Prediction Tickets")
        must_include_pos = st.multiselect("Numbers to include", options=list(range(1,50)), default=[])
        num_pos_pred_tickets = st.slider("Position-based tickets",1,10,3)

        def generate_position_based_ticket(position_most_common, must_include=[]):
            ticket = list({num for num,_ in position_most_common.values()})
            for n in must_include:
                if n not in ticket:
                    ticket.append(n)
            while len(ticket)<6:
                candidate = random.randint(1,49)
                if candidate not in ticket:
                    ticket.append(candidate)
            while len(ticket)>6:
                ticket.remove(random.choice([n for n in ticket if n not in must_include]))
            return sorted(ticket)

        for i in range(num_pos_pred_tickets):
            st.write(f"Position-Based Ticket {i+1}: {generate_position_based_ticket(position_most_common, must_include_pos)}")

        # --- Check if a Draw Combination Has Already Appeared ---
        st.subheader("🔍 Check if a Draw Combination Has Already Appeared (All History)")
        user_draw = st.text_input("Enter 6 numbers separated by commas (e.g., 5,12,19,23,34,45):", key="check_draw_combo")
        if user_draw.strip():
            try:
                numbers_entered = [int(x.strip()) for x in user_draw.split(",")]
                if len(numbers_entered)!=6 or not all(1<=n<=49 for n in numbers_entered):
                    raise ValueError("Enter exactly 6 numbers between 1-49")
                user_numbers = tuple(sorted(numbers_entered))
                past_draws = [tuple(sorted(row)) for row in numbers_df.values.tolist()]
                matches_idx = [i for i,row in enumerate(numbers_df.values.tolist()) if tuple(sorted(row))==user_numbers]
                if matches_idx:
                    st.success(f"✅ Combination appeared {len(matches_idx)} time(s)")
                    st.write("Occurrences:")
                    for i in matches_idx:
                        st.write(f"- {df_used.iloc[i][df_used.columns[:6]].tolist()}")
                else:
                    st.error("❌ Combination never appeared")
            except Exception as e:
                st.error(f"⚠️ Invalid input: {e}")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
