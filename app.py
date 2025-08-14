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

def to_py_ticket(ticket):
    return tuple(sorted(int(x) for x in ticket))

def extract_numbers_and_bonus(df):
    required_main_cols = [
        "NUMBER DRAWN 1", "NUMBER DRAWN 2", "NUMBER DRAWN 3",
        "NUMBER DRAWN 4", "NUMBER DRAWN 5", "NUMBER DRAWN 6",
    ]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in required_main_cols):
        return None, None, None

    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)
    dates = None
    if date_col:
        import re
        def clean_date_str(date_str):
            if pd.isna(date_str):
                return date_str
            return re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', str(date_str))
        df[date_col] = df[date_col].apply(clean_date_str)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        dates = df[date_col]

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df):
    all_numbers = numbers_df.values.flatten()
    return Counter(all_numbers)

@st.cache_data
def compute_pair_frequencies(numbers_df, limit=500):
    pair_counts = Counter()
    df_subset = numbers_df.tail(limit)
    for _, row in df_subset.iterrows():
        pairs = combinations(sorted(row.values), 2)
        pair_counts.update(pairs)
    return pair_counts

@st.cache_data
def compute_triplet_frequencies(numbers_df, limit=500):
    triplet_counts = Counter()
    df_subset = numbers_df.tail(limit)
    for _, row in df_subset.iterrows():
        triplets = combinations(sorted(row.values), 3)
        triplet_counts.update(triplets)
    return triplet_counts

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {num: -1 for num in range(1, 50)}
    gaps = {num: None for num in range(1, 50)}

    if dates is not None:
        numbers_df = numbers_df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for n in row.values:
            last_seen[n] = idx

    total_draws = len(numbers_df)
    for num in range(1, 50):
        gaps[num] = total_draws - 1 - last_seen[num] if last_seen[num] != -1 else total_draws
    return gaps

def most_common_per_draw_position(numbers_df):
    most_common_dict = {}
    for col in numbers_df.columns:
        counts = Counter(numbers_df[col])
        most_common_num, freq = counts.most_common(1)[0]
        most_common_dict[col] = (int(most_common_num), freq)
    return most_common_dict

# --- Main App ---

uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to 6, BONUS NUMBER, and optional DATE"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Find and parse date column
        date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(by=date_col, ascending=True)  # oldest first

        # Show all data uploaded
        st.subheader(f"Uploaded Data (All {len(df)} draws):")
        st.dataframe(df.reset_index(drop=True))

        # Slider for how many recent draws to analyze
        max_draws = len(df)
        st.subheader("Select Number of Recent Draws for Analysis")
        draws_to_use = st.slider("Number of draws", min_value=50, max_value=max_draws, value=min(300, max_draws), step=10)

        # Use last draws_to_use rows for analysis
        df_used = df.tail(draws_to_use).reset_index(drop=True)

        # Extract numbers & bonus
        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df_used)
        if numbers_df is None:
            st.error("CSV must have valid columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        # Frequencies and stats
        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]
        gaps = compute_number_gaps(numbers_df, dates)
        position_most_common = most_common_per_draw_position(numbers_df)

        # Display Hot & Cold
        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))
        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        # Most Common Numbers by Draw Position
        st.subheader("Most Common Numbers by Draw Position:")
        for position, (num, freq) in position_most_common.items():
            st.write(f"{position}: {num} (appeared {freq} times)")

        # Frequency Chart
        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter.get(x, 0))
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                     title=f"Number Frequency (last {draws_to_use} draws)", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        # Pair Frequency Chart
        st.subheader(f"Number Pair Frequency (last {draws_to_use} draws)")
        pair_counts = compute_pair_frequencies(numbers_df, limit=draws_to_use)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"])\
            .sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count",
                           color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        # Triplet Frequency Chart
        st.subheader(f"Number Triplet Frequency (last {draws_to_use} draws)")
        triplet_counts = compute_triplet_frequencies(numbers_df, limit=draws_to_use)
        triplets_df = pd.DataFrame(triplet_counts.items(), columns=["Triplet", "Count"])\
            .sort_values(by="Count", ascending=False).head(20)
        triplets_df["Triplet"] = triplets_df["Triplet"].apply(lambda x: f"{x[0]} & {x[1]} & {x[2]}")
        fig_triplets = px.bar(triplets_df, y="Triplet", x="Count", orientation='h', color="Count",
                             color_continuous_scale="Cividis")
        fig_triplets.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_triplets, use_container_width=True)

        # Gap Analysis
        st.subheader("Number Gap Analysis")
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())})\
            .sort_values(by="Gap", ascending=False)
        threshold = st.slider("Gap threshold for overdue numbers (draws)", min_value=0, max_value=100, value=27)
        st.dataframe(gaps_df[gaps_df["Gap"] >= threshold])

        # --- User Played Numbers Section ---
st.subheader("📊 Enter Your Played Numbers")
num_played_tickets = st.number_input("How many tickets have you played?", min_value=1, max_value=10, value=1, step=1)

user_played_tickets = []
for i in range(num_played_tickets):
    ticket_input = st.text_input(f"Ticket {i+1} (enter 6 numbers separated by commas)", "")
    if ticket_input:
        try:
            ticket_numbers = sorted([int(x.strip()) for x in ticket_input.split(",") if 1 <= int(x.strip()) <= 49])
            if len(ticket_numbers) != 6:
                st.warning(f"Ticket {i+1} does not have exactly 6 valid numbers.")
            else:
                user_played_tickets.append(ticket_numbers)
        except ValueError:
            st.warning(f"Ticket {i+1} contains invalid numbers.")

# --- Update Frequency with User Numbers ---
if user_played_tickets:
    st.subheader("Frequency Including Your Played Numbers")
    # Flatten previous numbers and user numbers
    combined_numbers = list(numbers_df.values.flatten()) + [n for ticket in user_played_tickets for n in ticket]
    combined_counter = Counter(combined_numbers)
    
    # Hot & Cold including user numbers
    hot_user = [num for num, _ in combined_counter.most_common(6)]
    cold_user = [num for num, _ in combined_counter.most_common()[:-7:-1]]
    
    st.write("Hot Numbers (Including Your Played Numbers):", hot_user)
    st.write("Cold Numbers (Including Your Played Numbers):", cold_user)

    # Optional: Highlight your numbers in generated tickets
    st.subheader("🎰 Highlighting Your Numbers in Generated Tickets")
    for idx, ticket in enumerate(generated_tickets, 1):
        highlighted = ["**" + str(n) + "**" if any(n in t for t in user_played_tickets) else str(n) for n in ticket]
        st.write(f"Ticket {idx}: {highlighted}")

        # Ticket Generator
        st.subheader("🎟️ Generate Lotto Tickets")
        strategy = st.selectbox(
            "Strategy for ticket generation",
            ["Pure Random", "Bias: Hot", "Bias: Cold", "Bias: Overdue", "Mixed"]
        )
        num_tickets = st.slider("How many tickets do you want to generate?", 1, 10, 5)

        def generate_ticket(pool):
            return sorted(random.sample(pool, 6))

        generated_tickets = []
        for _ in range(num_tickets):
            if strategy == "Pure Random":
                pool = list(range(1, 50))
            elif strategy == "Bias: Hot":
                pool = hot + random.sample([n for n in range(1, 50) if n not in hot], 43)
            elif strategy == "Bias: Cold":
                pool = cold + random.sample([n for n in range(1, 50) if n not in cold], 43)
            elif strategy == "Bias: Overdue":
                sorted_by_gap = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
                top_gap = [n for n, g in sorted_by_gap[:10]]
                pool = top_gap + random.sample([n for n in range(1, 50) if n not in top_gap], 39)
            elif strategy == "Mixed":
                pool = hot[:3] + cold[:2]
                pool += random.sample([n for n in range(1, 50) if n not in pool], 49 - len(pool))
            else:
                pool = list(range(1, 50))
            ticket = generate_ticket(pool)
            generated_tickets.append([int(n) for n in ticket])

        st.write("🎰 Your Generated Tickets:")
        for idx, ticket in enumerate(generated_tickets, 1):
            st.write(f"Ticket {idx}: {ticket}")

        # ML-based Prediction (existing)
        st.subheader("🧠 ML-Based Prediction (Experimental)")

        must_include = st.multiselect(
            "Select numbers you want to include in every ML ticket",
            options=list(range(1, 50)),
            default=[]
        )

        num_ml_tickets = st.slider("How many ML predicted tickets to generate?", 1, 10, 3)

        predicted_numbers = [int(num) for num, _ in counter.most_common(12)]

        st.write("Base Predicted Numbers (top 12 most common):")
        st.write(predicted_numbers)

        def generate_ml_ticket(must_include, predicted_numbers):
            ticket = must_include.copy()
            pool = [n for n in predicted_numbers if n not in ticket]
            needed = 6 - len(ticket)

            if len(pool) >= needed:
                ticket += random.sample(pool, needed)
            else:
                ticket += pool
                remaining_needed = 6 - len(ticket)
                remaining_pool = [n for n in range(1, 50) if n not in ticket]
                if remaining_needed > 0:
                    ticket += random.sample(remaining_pool, remaining_needed)

            swap_count = random.randint(1, 2)
            for _ in range(swap_count):
                idx_to_swap = random.randint(0, 5)
                if ticket[idx_to_swap] in must_include:
                    continue
                available_nums = [n for n in range(1, 50) if n not in ticket]
                if not available_nums:
                    break
                new_num = random.choice(available_nums)
                ticket[idx_to_swap] = new_num

            return sorted(int(n) for n in ticket)

        st.write("Generated ML Tickets:")
        for i in range(num_ml_tickets):
            ml_ticket = generate_ml_ticket(must_include, predicted_numbers)
            st.write(f"ML Ticket {i+1}: {ml_ticket}")

        # Position-Based Prediction Model (new)
        st.subheader("🎯 Position-Based Prediction Tickets")

        must_include_pos = st.multiselect(
            "Select numbers to always include in Position-Based Prediction",
            options=list(range(1, 50)),
            default=[]
        )

        num_pos_pred_tickets = st.slider("How many Position-Based predicted tickets to generate?", 1, 10, 3)

        def generate_position_based_ticket(position_most_common, must_include=[]):
            ticket = []
            # pick most common number in each position (column)
            for pos in sorted(position_most_common.keys()):
                num = position_most_common[pos][0]
                ticket.append(num)
            ticket = list(set(ticket))  # unique numbers only

            # Add must_include numbers if missing
            for n in must_include:
                if n not in ticket:
                    ticket.append(n)

            # Fill to 6 numbers randomly if needed
            while len(ticket) < 6:
                candidate = random.randint(1, 49)
                if candidate not in ticket:
                    ticket.append(candidate)

            # If more than 6 (due to must_include), randomly remove extras except must_include
            while len(ticket) > 6:
                removable = [n for n in ticket if n not in must_include]
                if not removable:
                    break
                to_remove = random.choice(removable)
                ticket.remove(to_remove)

            # Random swaps for variation (excluding must_include)
            swap_count = random.randint(1, 2)
            for _ in range(swap_count):
                idx = random.randint(0, 5)
                if ticket[idx] in must_include:
                    continue
                available = [n for n in range(1, 50) if n not in ticket]
                if not available:
                    break
                ticket[idx] = random.choice(available)

            return sorted(ticket)

        st.write("Position-Based Predicted Tickets:")
        for i in range(num_pos_pred_tickets):
            pos_ticket = generate_position_based_ticket(position_most_common, must_include_pos)
            st.write(f"Position-Based Ticket {i+1}: {pos_ticket}")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
