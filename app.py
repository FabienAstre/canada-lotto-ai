import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
import plotly.express as px
from scipy.stats import chisquare

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

    # Validate main columns
    if not all(col in df.columns for col in required_main_cols):
        return None, None, None

    # Extract and validate main numbers
    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    # Extract and validate bonus number
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    # Identify and clean the date column
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

# Most Common Number per Draw Position
def most_common_per_draw_position(numbers_df):
    most_common_dict = {}
    for col in numbers_df.columns:
        counts = Counter(numbers_df[col])
        most_common_num, freq = counts.most_common(1)[0]
        most_common_dict[col] = (int(most_common_num), freq)
    return most_common_dict

# Number streaks: count how often each number appeared in consecutive draws
def compute_streaks(numbers_df):
    streaks = {num: 0 for num in range(1, 50)}
    max_streaks = {num: 0 for num in range(1, 50)}
    prev_draw = set()
    for _, row in numbers_df.iterrows():
        current_draw = set(row.values)
        for num in range(1, 50):
            if num in current_draw and num in prev_draw:
                streaks[num] += 1
                max_streaks[num] = max(max_streaks[num], streaks[num])
            else:
                streaks[num] = 0
        prev_draw = current_draw
    return max_streaks

# Bad pattern checks
def is_bad_pattern(ticket):
    # All even or all odd
    if all(n % 2 == 0 for n in ticket) or all(n % 2 == 1 for n in ticket):
        return True
    # Sequential numbers (length >=3)
    seq_count = 1
    for i in range(1, len(ticket)):
        if ticket[i] == ticket[i-1] + 1:
            seq_count += 1
            if seq_count >= 3:
                return True
        else:
            seq_count = 1
    # All numbers in one 10-number range
    ranges = [(1,10),(11,20),(21,30),(31,40),(41,49)]
    for r in ranges:
        if all(r[0] <= n <= r[1] for n in ticket):
            return True
    return False

# Weighted random sampling for ticket generation
def generate_weighted_ticket(freq_counter, k=6, exclude_bad_patterns=True):
    numbers = list(range(1, 50))
    frequencies = np.array([freq_counter.get(n, 0) for n in numbers], dtype=float)
    weights = frequencies + 1  # avoid zeros
    weights /= weights.sum()
    attempt = 0
    while True:
        attempt += 1
        ticket = np.random.choice(numbers, size=k, replace=False, p=weights)
        ticket = sorted(ticket)
        if exclude_bad_patterns and is_bad_pattern(ticket):
            if attempt > 100:
                # fallback if too many attempts
                break
            continue
        break
    return ticket

# Recency weighting: decay weights for older draws
def compute_recency_weights(numbers_df):
    n_draws = len(numbers_df)
    decay_factor = 0.95  # exponential decay per draw (adjustable)
    weights = {num: 0.0 for num in range(1, 50)}

    for idx, row in numbers_df.iterrows():
        weight = decay_factor ** (n_draws - idx - 1)
        for num in row.values:
            weights[num] += weight

    # Normalize weights
    total = sum(weights.values())
    for num in weights:
        weights[num] /= total
    return weights

# Bonus number frequency and gap analysis
def analyze_bonus(bonus_series):
    if bonus_series is None or bonus_series.empty:
        return None, None
    bonus_freq = Counter(bonus_series)
    last_seen = {num: -1 for num in range(1, 50)}
    for idx, num in enumerate(bonus_series):
        last_seen[num] = idx
    total = len(bonus_series)
    bonus_gap = {num: total - 1 - last_seen[num] if last_seen[num] != -1 else total for num in range(1, 50)}
    return bonus_freq, bonus_gap

# Chi-square test for uniformity of number frequencies
def perform_chi_square_test(freq_counter, total_draws):
    observed = np.array([freq_counter.get(n, 0) for n in range(1, 50)])
    expected = np.full(49, total_draws * 6 / 49)  # Expected frequency if uniform
    chi2, p_value = chisquare(observed, expected)
    return chi2, p_value

# Plot frequency trend over time with moving average
def plot_frequency_trends(numbers_df):
    freq_over_time = pd.DataFrame(0, index=range(len(numbers_df)), columns=range(1,50))
    for idx, row in numbers_df.iterrows():
        for num in row.values:
            freq_over_time.at[idx, num] = 1
    freq_cumsum = freq_over_time.cumsum()
    window = 20
    freq_mavg = freq_cumsum.diff(window).fillna(0)

    fig = px.line(freq_mavg, title=f"Frequency Moving Average (window={window} draws)")
    fig.update_layout(xaxis_title="Draw Number (Oldest to Newest)", yaxis_title="Frequency")
    return fig

# --- App Main ---

uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to 6, BONUS NUMBER, and optional DATE"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Find date column
        date_col = next((col for col in ['DATE', 'Draw Date', 'Draw_Date', 'Date'] if col in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(by=date_col, ascending=True)  # Oldest first for trend analysis

        columns_to_display = df.columns.tolist()
        if date_col and date_col not in columns_to_display:
            columns_to_display.insert(0, date_col)

        st.subheader("Uploaded Data (Last 300 draws):")
        st.dataframe(df.tail(300)[columns_to_display].reset_index(drop=True))

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("CSV must have valid columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]
        gaps = compute_number_gaps(numbers_df, dates)

        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))

        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        # Most common numbers per draw position
        position_most_common = most_common_per_draw_position(numbers_df)
        st.subheader("Most Common Numbers by Draw Position:")
        for position, (num, freq) in position_most_common.items():
            st.write(f"{position}: {num} (appeared {freq} times)")

        # Frequency Chart
        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter.get(x, 0))
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                     title="Number Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        # Pair Frequency
        st.subheader("Number Pair Frequency (last 500 draws)")
        pair_counts = compute_pair_frequencies(numbers_df)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"])\
            .sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count",
                           color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        # Triplet Frequency
        st.subheader("Number Triplet Frequency (last 500 draws)")
        triplet_counts = compute_triplet_frequencies(numbers_df)
        triplets_df = pd.DataFrame(triplet_counts.items(), columns=["Triplet", "Count"])\
            .sort_values(by="Count", ascending=False).head(10)
        triplets_df["Triplet"] = triplets_df["Triplet"].apply(lambda x: f"{x[0]}, {x[1]} & {x[2]}")
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

        # Number streaks
        st.subheader("Number Streaks (Max Consecutive Draws)")
        streaks = compute_streaks(numbers_df)
        streaks_df = pd.DataFrame({"Number": list(streaks.keys()), "Max Streak": list(streaks.values())})\
            .sort_values(by="Max Streak", ascending=False)
        st.dataframe(streaks_df.head(15))

        # Bonus number analysis
        st.subheader("Bonus Number Analysis")
        bonus_freq, bonus_gap = analyze_bonus(bonus_series)
        if bonus_freq:
            bonus_freq_df = pd.DataFrame({"Number": list(bonus_freq.keys()), "Frequency": list(bonus_freq.values())})\
                .sort_values(by="Frequency", ascending=False)
            st.write("Bonus Number Frequencies:")
            st.dataframe(bonus_freq_df.head(10))

            bonus_gap_df = pd.DataFrame({"Number": list(bonus_gap.keys()), "Gap": list(bonus_gap.values())})\
                .sort_values(by="Gap", ascending=False)
            st.write("Bonus Number Gap Analysis:")
            st.dataframe(bonus_gap_df.head(10))
        else:
            st.write("No valid bonus number data found.")

        # Frequency trend over time
        st.subheader("Frequency Trend Over Time (Moving Average)")
        fig_trend = plot_frequency_trends(numbers_df)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Statistical uniformity test
        st.subheader("Chi-Square Test for Uniformity")
        chi2, p_val = perform_chi_square_test(counter, len(numbers_df))
        st.write(f"Chi-square statistic: {chi2:.2f}")
        st.write(f"P-value: {p_val:.4f} (higher means numbers appear more uniformly)")

        # Ticket Generator
        st.subheader("🎟️ Generate Lotto Tickets")

        strategy = st.selectbox(
            "Strategy for ticket generation",
            ["Pure Random", "Bias: Hot", "Bias: Cold", "Bias: Overdue", "Mixed", "Weighted Frequencies"]
        )
        num_tickets = st.slider("How many tickets do you want to generate?", 1, 10, 5)

        # Recency weighting option
        use_recency = st.checkbox("Use Recency Weighted Frequencies?", value=False)
        recency_weights = compute_recency_weights(numbers_df) if use_recency else None

        def generate_ticket(pool):
            return sorted(random.sample(pool, 6))

        generated_tickets = []
        for _ in range(num_tickets):
            if strategy == "Pure Random":
                pool = list(range(1, 50))
                ticket = generate_ticket(pool)

            elif strategy == "Bias: Hot":
                pool = hot + random.sample([n for n in range(1, 50) if n not in hot], 43)
                ticket = generate_ticket(pool)

            elif strategy == "Bias: Cold":
                pool = cold + random.sample([n for n in range(1, 50) if n not in cold], 43)
                ticket = generate_ticket(pool)

            elif strategy == "Bias: Overdue":
                sorted_by_gap = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
                top_gap = [n for n, g in sorted_by_gap[:10]]
                pool = top_gap + random.sample([n for n in range(1, 50) if n not in top_gap], 39)
                ticket = generate_ticket(pool)

            elif strategy == "Mixed":
                pool = hot[:3] + cold[:2]
                pool += random.sample([n for n in range(1, 50) if n not in pool], 49 - len(pool))
                ticket = generate_ticket(pool)

            elif strategy == "Weighted Frequencies":
                # Use recency weights if checked, else use frequency counter
                if recency_weights:
                    weights = np.array([recency_weights[n] for n in range(1, 50)])
                    numbers = list(range(1, 50))
                    ticket = []
                    attempt = 0
                    while len(ticket) < 6 and attempt < 100:
                        attempt += 1
                        candidate = np.random.choice(numbers, p=weights)
                        if candidate not in ticket:
                            ticket.append(candidate)
                    ticket = sorted(ticket)
                    # Avoid bad patterns
                    if is_bad_pattern(ticket):
                        ticket = generate_weighted_ticket(counter)
                else:
                    ticket = generate_weighted_ticket(counter)

            else:
                pool = list(range(1, 50))
                ticket = generate_ticket(pool)

            generated_tickets.append(ticket)

        st.write("🎰 Your Generated Tickets:")
        for idx, ticket in enumerate(generated_tickets, 1):
            st.write(f"Ticket {idx
