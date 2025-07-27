import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- Streamlit Setup ---
st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Upload Lotto 6/49 CSV file with columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' and 'BONUS NUMBER'")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# --- Helper functions ---

def extract_numbers_and_bonus(df):
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"
    if not all(c in df.columns for c in main_cols):
        return None, None

    main_df = df[main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    return main_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None


def calculate_gaps(numbers_df):
    last_seen = {n: None for n in range(1, 50)}
    for idx, row in enumerate(reversed(numbers_df.values.tolist())):
        for num in row:
            if last_seen[num] is None:
                last_seen[num] = idx
    return {n: (last_seen[n] if last_seen[n] is not None else len(numbers_df)) for n in range(1, 50)}


def generate_tickets(numbers_pool, n_tickets=5):
    tickets = []
    for _ in range(n_tickets):
        if len(numbers_pool) >= 6:
            ticket = sorted(random.sample(numbers_pool, 6))
        else:
            ticket = sorted(random.sample(range(1, 50), 6))
        bonus_pool = [n for n in range(1, 50) if n not in ticket]
        bonus = random.choice(bonus_pool)
        tickets.append((tuple(ticket), bonus))
    return tickets


def random_forest_prediction(numbers_df, n_draws=100):
    df = numbers_df.tail(n_draws)
    all_numbers = list(range(1, 50))

    freq_features = []
    targets = []
    window_size = 10

    for i in range(window_size, len(df)):
        window = df.iloc[i - window_size:i]
        freq = Counter(window.values.flatten())
        feature = [freq.get(num, 0) for num in all_numbers]
        freq_features.append(feature)
        targets.append(df.iloc[i].tolist())

    if not freq_features:
        return []

    X = np.array(freq_features)
    y_list = targets
    rf_scores = np.zeros(49)

    for num in all_numbers:
        y = np.array([1 if num in draw else 0 for draw in y_list])
        if y.sum() == 0:
            continue
        model_rf = RandomForestClassifier(n_estimators=100)
        model_rf.fit(X, y)
        prob = model_rf.predict_proba([X[-1]])[0][1]
        rf_scores[num - 1] = prob

    top_predicted = [i + 1 for i in np.argsort(rf_scores)[-15:][::-1]]
    return top_predicted


# --- Main ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        numbers_df, bonus_series = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("CSV must have valid 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' columns with numbers 1-49.")
            st.stop()

        st.subheader("All Imported Draws:")
        st.dataframe(numbers_df)

        # Display last 30 draws
        st.subheader("Last 30 Draws:")
        st.dataframe(numbers_df.tail(30).reset_index(drop=True))
        if bonus_series is not None:
            st.subheader("Last 30 Bonus Numbers:")
            st.write(bonus_series.tail(30).tolist())

        # Frequency analysis
        all_nums = numbers_df.values.flatten()
        counter = Counter(all_nums)
        bonus_counter = Counter(bonus_series) if bonus_series is not None else Counter()

        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]

        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))
        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        if bonus_series is not None:
            st.subheader("Most Frequent Bonus Numbers:")
            bonus_hot = [num for num, _ in bonus_counter.most_common(6)]
            st.write(", ".join(map(str, bonus_hot)))

        # Frequency Bar Chart
        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter.get(x, 0))

        fig = px.bar(
            freq_df,
            x="Number",
            y="Frequency",
            title="Number Frequencies",
            labels={"Number": "Number", "Frequency": "Count"},
            color="Frequency",
            color_continuous_scale="Blues"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Pair frequency
        pair_counts = Counter()
        for _, row in numbers_df.iterrows():
            pairs = combinations(sorted(row.values), 2)
            pair_counts.update(pairs)

        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"]).sort_values(by="Count", ascending=False)

        fig_pairs = px.bar(
            pairs_df.head(20),
            y=pairs_df["Pair"].astype(str),
            x="Count",
            orientation='h',
            title="Pair Number Frequencies (Top 20)",
            labels={"Count": "Count", "Pair": "Pair"},
            color="Count",
            color_continuous_scale="Viridis"
        )
        fig_pairs.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white")
        st.plotly_chart(fig_pairs, use_container_width=True)

        # Number Gap Analysis
        gap_threshold = st.slider("Gap threshold for overdue numbers (draws)", 0, 100, 20)
        gaps = calculate_gaps(numbers_df)
        overdue = sorted([(num, gap) for num, gap in gaps.items() if gap >= gap_threshold], key=lambda x: x[1], reverse=True)

        st.subheader(f"Overdue Numbers (gap ≥ {gap_threshold} draws):")
        if overdue:
            st.dataframe(pd.DataFrame(overdue, columns=["Number", "Gap (draws)"]))
        else:
            st.write("No overdue numbers at this threshold.")

        # Prediction & Smart Ticket Generation
        st.subheader("Prediction & Smart Ticket Generation")

        if st.button("Run Random Forest Prediction on last 100 draws"):
            st.info("Training Random Forest... Please wait.")
            top_pred = random_forest_prediction(numbers_df, n_draws=100)
            if not top_pred:
                st.warning("Not enough data to train prediction.")
            else:
                st.write("Top Predicted Numbers (Random Forest):")
                st.write(", ".join(map(str, top_pred[:10])))

                # Combine pools for smart ticket generation
                pool = list(set(hot + cold + top_pred[:10] + [num for num, _ in overdue[:5]]))

                st.write(f"Smart Pool (size {len(pool)}): ", pool)

                n_smart_tickets = st.slider("Number of Smart Tickets to generate", 1, 20, 5)
                smart_tickets = generate_tickets(pool, n_tickets=n_smart_tickets)

                st.subheader("Smart Generated Tickets (Main + Bonus):")
                for i, (main_nums, bonus_num) in enumerate(smart_tickets, 1):
                    st.write(f"{i}: Main: {main_nums} | Bonus: {bonus_num}")

        # Hot/Cold Ticket Generation
        st.subheader("Hot/Cold Ticket Generation")

        budget = st.slider("Budget ($)", min_value=3, max_value=300, value=30, step=3)
        price_per_ticket = 3
        n_tickets = budget // price_per_ticket

        if st.button("Generate Hot/Cold Tickets"):
            tickets = generate_tickets(hot + cold, n_tickets=n_tickets)
            st.subheader(f"Generated {len(tickets)} Hot/Cold Tickets (Main + Bonus):")
            for i, (main_nums, bonus_num) in enumerate(tickets, 1):
                st.write(f"{i}: Main: {main_nums} | Bonus: {bonus_num}")

    except Exception as e:
        st.error(f"Error loading or processing CSV: {e}")

else:
    st.info("Please upload a CSV file with the required columns to start.")
