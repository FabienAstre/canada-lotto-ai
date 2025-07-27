import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Upload your Lotto 6/49 CSV file to analyze draws and generate tickets.")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Import Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to NUMBER DRAWN 6 and BONUS NUMBER",
)

# --- Helper Functions ---
@st.cache_data
def extract_numbers_and_bonus(df):
    """Extract main numbers and bonus from CSV."""
    required_main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in required_main_cols):
        return None, None

    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None


def generate_tickets(hot, cold, n_tickets):
    """Generate tickets based on hot/cold numbers."""
    tickets = set()
    pool = 49
    main_needed = 6

    while len(tickets) < n_tickets:
        n_hot = random.randint(2, min(4, len(hot)))
        n_cold = random.randint(2, min(4, len(cold)))

        pick_hot = random.sample(hot, n_hot)
        pick_cold = random.sample(cold, n_cold)

        main_numbers = set(pick_hot + pick_cold)
        while len(main_numbers) < main_needed:
            main_numbers.add(random.randint(1, pool))

        main_numbers = tuple(sorted(main_numbers))
        bonus_pool = set(range(1, pool + 1)) - set(main_numbers)
        bonus_number = random.choice(list(bonus_pool))

        ticket = main_numbers + (bonus_number,)
        tickets.add(ticket)

    return list(tickets)


# --- Main Logic ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data (All Draws):")
        st.dataframe(df)

        numbers_df, bonus_series = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("CSV must contain valid columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with numbers 1-49.")
        else:
            st.subheader("Recent Draws (Last 30):")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Recent Bonus Numbers (Last 30):")
                st.write(bonus_series.tail(30).to_list())

            # --- Frequency Analysis ---
            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)
            hot = [num for num, _ in counter.most_common(6)]
            cold = [num for num, _ in counter.most_common()[:-7:-1]]

            st.subheader("Hot Numbers:")
            st.write(", ".join(map(str, hot)))
            st.subheader("Cold Numbers:")
            st.write(", ".join(map(str, cold)))

            # --- Frequency Chart ---
            freq_df = pd.DataFrame({"Number": list(range(1, 50))})
            freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter[x] if x in counter else 0)

            fig = px.bar(
                freq_df,
                x="Number",
                y="Frequency",
                title="Frequency of Numbers (All Draws)",
                color="Frequency",
                color_continuous_scale="Blues",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # --- Pair Frequency Chart (last 300 draws for speed) ---
            pair_counts = Counter()
            limited_df = numbers_df.tail(300)
            for _, row in limited_df.iterrows():
                pairs = combinations(sorted(row.values), 2)
                pair_counts.update(pairs)

            pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"])
            pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
            pairs_df = pairs_df.sort_values(by="Count", ascending=False).head(20)

            fig_pairs = px.bar(
                pairs_df,
                y="Pair",
                x="Count",
                orientation='h',
                title="Number Pair Frequency (Last 300 Draws)",
                color="Count",
                color_continuous_scale="Viridis",
            )
            fig_pairs.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_pairs, use_container_width=True)

            # --- Ticket Generation ---
            st.subheader("Ticket Generation")
            budget = st.slider("Budget ($)", min_value=3, max_value=300, value=30, step=3)
            price_per_ticket = 3
            n_tickets = budget // price_per_ticket

            if st.button("Generate Tickets"):
                tickets = generate_tickets(hot, cold, n_tickets)
                st.subheader(f"Generated Tickets ({len(tickets)}):")
                for i, t in enumerate(tickets, 1):
                    st.write(f"{i}: Main Numbers: {t[:-1]} | Bonus: {t[-1]}")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.info("Please upload a CSV file to start analysis.")
