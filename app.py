import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from itertools import combinations
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
# File Upload
# -------------------
uploaded_file = st.file_uploader("Upload a Lotto 6/49 CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of uploaded data")
        st.dataframe(df.head())

        # -------------------
        # Extract Numbers & Dates
        # -------------------
        main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
        bonus_col = "BONUS NUMBER"
        date_col = next((c for c in ['DATE','Draw Date','Draw_Date','Date'] if c in df.columns), None)

        numbers_df = df[main_cols].apply(pd.to_numeric, errors='coerce').dropna()
        bonus_series = None
        if bonus_col in df.columns:
            bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        dates = pd.to_datetime(df[date_col], errors='coerce') if date_col else None

        # -------------------
        # Slider: number of draws to analyze
        # -------------------
        max_draws = st.slider("Number of past draws to analyze", min_value=50, max_value=len(numbers_df), value=200)
        draws_to_use = numbers_df.head(max_draws).values.tolist()

        # -------------------
        # Helper Functions
        # -------------------
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

        def compute_number_gaps(numbers_df):
            last_seen = {n: -1 for n in range(1,50)}
            df_sorted = numbers_df.reset_index(drop=True)
            for idx, row in df_sorted.iterrows():
                for n in row.values:
                    last_seen[n] = idx
            total_draws = len(df_sorted)
            gaps = {n: total_draws - 1 - last_seen[n] if last_seen[n] != -1 else total_draws for n in range(1,50)}
            return gaps

        def most_common_per_position(numbers_df):
            result = {}
            for col in numbers_df.columns:
                num, freq = Counter(numbers_df[col]).most_common(1)[0]
                result[col] = (int(num), int(freq))
            return result

        # -------------------
        # Ticket Generators
        # -------------------
        def generate_ticket(pool):
            return sorted(random.sample(pool, 6))

        def generate_balanced_ticket():
            while True:
                low_pool = list(range(1,17))
                mid_pool = list(range(17,34))
                high_pool = list(range(34,50))
                ticket = random.sample(low_pool,2) + random.sample(mid_pool,2) + random.sample(high_pool,2)
                ticket = sorted(ticket)
                odds = sum(1 for n in ticket if n%2==1)
                evens = 6 - odds
                total = sum(ticket)
                if odds==3 and evens==3 and 100 <= total <=180:
                    return ticket

        def generate_delta_tickets(draws, n_tickets=6):
            tickets = []
            deltas = []
            for draw in draws:
                sorted_draw = sorted(draw)
                deltas.extend(np.diff(sorted_draw))
            common_deltas = [d for d,_ in Counter(deltas).most_common(10)]
            for _ in range(n_tickets):
                start = random.randint(1,10)
                seq = [start]
                for _ in range(5):
                    delta = random.choice(common_deltas)
                    next_num = seq[-1]+delta
                    if next_num<=49:
                        seq.append(next_num)
                seq = sorted(set(seq))[:6]
                tickets.append([int(x) for x in seq])
            return tickets

        def generate_zone_tickets(n_tickets=6):
            zones = [(1,16),(17,33),(34,49)]
            tickets = []
            for _ in range(n_tickets):
                ticket = [random.randint(*z) for z in zones]
                while len(ticket)<6:
                    candidate = random.randint(1,49)
                    if candidate not in ticket:
                        ticket.append(candidate)
                tickets.append(sorted(ticket))
            return tickets

        # -------------------
        # Strategy Tabs with Explanations
        # -------------------
        st.subheader("🎟️ Ticket Generators")
        strategy = st.selectbox("Choose Ticket Generation Strategy", [
            "Pure Random", "Balanced", "Delta System", "Zone Coverage"
        ])

        num_tickets = st.slider("Number of tickets to generate", 1, 10, 6)

        tickets=[]
        if strategy=="Pure Random":
            st.markdown("Randomly selects 6 numbers from 1–49.")
            for _ in range(num_tickets):
                tickets.append(generate_ticket(list(range(1,50))))
        elif strategy=="Balanced":
            st.markdown("Balanced ticket: 3 odd & 3 even numbers, 2 from low/mid/high pools, sum 100–180.")
            for _ in range(num_tickets):
                tickets.append(generate_balanced_ticket())
        elif strategy=="Delta System":
            st.markdown("Δ Delta System: Builds tickets from most common differences (gaps) in historical draws.")
            tickets = generate_delta_tickets(draws_to_use, n_tickets=num_tickets)
        elif strategy=="Zone Coverage":
            st.markdown("Ensures coverage from each zone: Low (1–16), Mid (17–33), High (34–49).")
            tickets = generate_zone_tickets(num_tickets)

        st.subheader(f"Generated {num_tickets} Tickets")
        for i,t in enumerate(tickets,1):
            st.write(f"Ticket {i}: {t}")

        # -------------------
        # Frequency & Gap Analysis
        # -------------------
        st.subheader("Number Frequency Analysis")
        counter = compute_frequencies(numbers_df)
        freq_df = pd.DataFrame({"Number": range(1,50), "Frequency":[counter.get(n,0) for n in range(1,50)]})
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Number Gap Analysis")
        gaps = compute_number_gaps(numbers_df)
        gaps_df = pd.DataFrame({"Number":list(gaps.keys()),"Gap":list(gaps.values())}).sort_values(by="Gap",ascending=False)
        st.table(gaps_df.head(10))

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
