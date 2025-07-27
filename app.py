import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---- Streamlit Setup ----
st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Upload your Lotto 6/49 CSV file to analyze draws, run predictions, and generate tickets.")

# ---- File Upload ----
uploaded_file = st.file_uploader(
    "Import Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to NUMBER DRAWN 6 and BONUS NUMBER",
)

# ---- Helper Functions ----
def extract_numbers_and_bonus(df):
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


def calculate_number_gaps(numbers_df):
    """Calculate how many draws since each number last appeared."""
    gaps = {n: 0 for n in range(1, 50)}
    for i, row in enumerate(reversed(numbers_df.values.tolist())):
        for n in range(1, 50):
            if n not in row and gaps[n] == 0:
                gaps[n] = i + 1
    return gaps


def generate_tickets(numbers_pool, n_tickets=5):
    """Generate tickets from a given pool of numbers."""
    tickets = []
    for _ in range(n_tickets):
        ticket = sorted(random.sample(numbers_pool, 6))
        bonus_pool = [n for n in range(1, 50) if n not in ticket]
        bonus = random.choice(bonus_pool)
        tickets.append((tuple(ticket), bonus))
    return tickets


def random_forest_prediction(numbers_df, n_draws=100):
    """Train a Random Forest model on last n_draws and predict top numbers."""
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
        next_probs_rf = model_rf.predict_proba([X[-1]])[0][1]
        rf_scores[num - 1] = next_probs_rf

    top_predicted = [i + 1 for i in np.argsort(rf_scores)[-15:][::-1]]
    return top_predicted


# ---- Main Logic ----
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        numbers_df, bonus_series = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("CSV must contain valid columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with numbers 1-49.")
        else:
            st.subheader("All Uploaded Draws:")
            st.dataframe(numbers_df)

            # Frequency Analysis
            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)
            hot = [num for num, _ in counter.most_common(6)]
            cold = [num for num, _ in counter.most_common()[:-7:-1]]

            st.subheader("Hot Numbers:")
            st.write(", ".join(map(str, hot)))
            st.subheader("Cold Numbers:")
            st.write(", ".join(map(str, cold)))

            # Number Gap Analysis
            gaps = calculate_number_gaps(numbers_df)
            overdue = sorted([(n, g) for n, g in gaps.items() if g >= 20], key=lambda x: x[1], reverse=True)
            if overdue:
                st.subheader("Overdue Numbers (gap ≥ 20 draws):")
                overdue_df = pd.DataFrame(overdue, columns=["Number", "Gap (draws)"])
                st.dataframe(overdue_df)

            # Random Forest Prediction
            if st.button("Run Random Forest Prediction"):
                st.info("Training Random Forest on last 100 draws...")
                top_predictions = random_forest_prediction(numbers_df, n_draws=100)
                if top_predictions:
                    st.subheader("Top Predicted Numbers (Random Forest):")
                    st.write(", ".join(map(str, top_predictions[:10])))

                    # Smart ticket generation
                    smart_pool = list(set(hot + cold + top_predictions[:10] + [n for n, _ in overdue[:5]]))
                    smart_tickets = generate_tickets(smart_pool, n_tickets=5)

                    st.subheader("Smart Tickets (Main + Bonus):")
                    for i, (main, bonus) in enumerate(smart_tickets, 1):
                        st.write(f"{i}: Main: {main} | Bonus: {bonus}")
                else:
                    st.warning("Not enough data to train Random Forest.")

            # Hot/Cold Ticket Generation
            st.subheader("Hot/Cold Ticket Generation")
            budget = st.slider("Budget ($)", min_value=3, max_value=300, value=30, step=3)
            price_per_ticket = 3
            n_tickets = budget // price_per_ticket

            if st.button("Generate Hot/Cold Tickets"):
                tickets = generate_tickets(hot + cold, n_tickets=n_tickets)
                for i, (main, bonus) in enumerate(tickets, 1):
                    st.write(f"{i}: Main: {main} | Bonus: {bonus}")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
else:
    st.info("Please upload a CSV file to start analysis.")
