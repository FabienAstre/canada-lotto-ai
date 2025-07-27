import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see predictions.")

def to_py_ticket(ticket):
    return tuple(sorted(int(x) for x in ticket))

def extract_numbers_and_bonus(df):
    required_main_cols = [
        "NUMBER DRAWN 1",
        "NUMBER DRAWN 2",
        "NUMBER DRAWN 3",
        "NUMBER DRAWN 4",
        "NUMBER DRAWN 5",
        "NUMBER DRAWN 6",
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

    date_col = None
    for col_candidate in ['DATE', 'Draw Date', 'Draw_Date', 'Date']:
        if col_candidate in df.columns:
            date_col = col_candidate
            break

    dates = None
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce')
        except:
            dates = None

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

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {num: -1 for num in range(1, 50)}
    gaps = {num: None for num in range(1, 50)}

    if dates is not None:
        order = dates.argsort()
        numbers_df = numbers_df.iloc[order].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for n in row.values:
            last_seen[n] = idx

    total_draws = len(numbers_df)
    for num in range(1, 50):
        gaps[num] = total_draws - 1 - last_seen[num] if last_seen[num] != -1 else total_draws
    return gaps

def generate_tickets_hot_cold(hot, cold, n_tickets):
    tickets = set()
    while len(tickets) < n_tickets:
        n_hot = random.randint(2, min(4, len(hot)))
        n_cold = random.randint(2, min(4, len(cold)))
        pick_hot = random.sample(hot, n_hot)
        pick_cold = random.sample(cold, n_cold)
        current = set(pick_hot + pick_cold)
        while len(current) < 6:
            current.add(random.randint(1, 49))
        tickets.add(to_py_ticket(current))
    return list(tickets)

def generate_tickets_weighted(counter, n_tickets):
    numbers = np.array(range(1, 50))
    freqs = np.array([counter.get(num, 0) for num in numbers])
    weights = freqs + 1
    tickets = set()
    while len(tickets) < n_tickets:
        ticket_np = np.random.choice(numbers, 6, replace=False, p=weights/weights.sum())
        tickets.add(to_py_ticket(ticket_np))
    return list(tickets)

def generate_smart_tickets(n_tickets, fixed_nums, exclude_nums, due_nums):
    tickets = set()
    pool = set(range(1, 50)) - exclude_nums - fixed_nums
    while len(tickets) < n_tickets:
        ticket = set(fixed_nums)
        for num in (due_nums - ticket):
            if len(ticket) < 6:
                ticket.add(num)
        remaining_pool = list(pool - ticket)
        random.shuffle(remaining_pool)
        for num in remaining_pool:
            if len(ticket) >= 6:
                break
            ticket.add(num)
        tickets.add(to_py_ticket(ticket))
    return list(tickets)

@st.cache_data
def build_prediction_features(numbers_df, max_draws=500):
    numbers_df = numbers_df.tail(max_draws).reset_index(drop=True)
    total_draws = len(numbers_df)
    df_feat = []
    last_seen_draw = {num: -1 for num in range(1, 50)}
    freq_counter = Counter()

    for idx, row in numbers_df.iterrows():
        current_numbers = set(row.values)
        for num in range(1, 50):
            gap = idx - last_seen_draw[num] if last_seen_draw[num] != -1 else total_draws
            freq = freq_counter[num]
            df_feat.append({
                "draw_index": idx,
                "number": num,
                "gap": gap,
                "frequency": freq,
                "appeared_next": None
            })
        for num in current_numbers:
            last_seen_draw[num] = idx
            freq_counter[num] += 1

    df_feat = pd.DataFrame(df_feat)

    df_feat['appeared_next'] = 0
    for idx in range(total_draws-1):
        appeared_next_vals = set(numbers_df.iloc[idx+1].values)
        mask = df_feat['draw_index'] == idx
        df_feat.loc[mask, 'appeared_next'] = df_feat.loc[mask, 'number'].apply(lambda x: 1 if x in appeared_next_vals else 0)

    return df_feat

@st.cache_data
def train_predictive_model(df_feat):
    feature_cols = ['gap', 'frequency']
    X = df_feat[feature_cols]
    y = df_feat['appeared_next']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

def predict_next_draw_probs(model, numbers_df):
    total_draws = len(numbers_df)
    last_seen_draw = {num: -1 for num in range(1, 50)}
    freq_counter = Counter()

    for idx, row in numbers_df.iterrows():
        current_numbers = set(row.values)
        for num in current_numbers:
            last_seen_draw[num] = idx
            freq_counter[num] += 1

    features = []
    for num in range(1, 50):
        gap = total_draws - 1 - last_seen_draw[num] if last_seen_draw[num] != -1 else total_draws
        freq = freq_counter[num]
        features.append([gap, freq])

    X_pred = pd.DataFrame(features, columns=['gap', 'frequency'])
    probs = model.predict_proba(X_pred)[:,1]
    return dict(zip(range(1, 50), probs))

uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to NUMBER DRAWN 6 and BONUS NUMBER",
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data (Last 30 draws):")
        st.dataframe(df.tail(30))

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("CSV must have columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]

        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))
        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter[x] if x in counter else 0)
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", title="Number Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Number Pair Frequency (last 500 draws)")
        pair_counts = compute_pair_frequencies(numbers_df, limit=500)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"]).sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count", color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        st.subheader("Number Gap Analysis")
        gaps = compute_number_gaps(numbers_df, dates)
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values(by="Gap", ascending=False)
        overdue_threshold = st.slider("Gap threshold for overdue numbers (draws)", min_value=0, max_value=100, value=27)
        st.dataframe(gaps_df[gaps_df["Gap"] >= overdue_threshold])

        if dates is not None:
            numbers_df_with_dates = numbers_df.copy()
            numbers_df_with_dates['Date'] = dates
            numbers_df_with_dates = numbers_df_with_dates.dropna(subset=['Date']).reset_index(drop=True)
            numbers_df_with_dates['Year'] = numbers_df_with_dates['Date'].dt.year

            years = sorted(numbers_df_with_dates['Year'].unique())
            numbers = list(range(1, 50))
            freq_matrix = pd.DataFrame(0, index=numbers, columns=years)

            for year in years:
                yearly_data = numbers_df_with_dates[numbers_df_with_dates['Year'] == year]
                nums = yearly_data.iloc[:, 0:6].values.flatten()
                counts = Counter(nums)
                for num in counts:
                    freq_matrix.at[num, year] = counts[num]

            st.subheader("Heatmap: Number Frequency by Year")
            fig_heat, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(freq_matrix, cmap="YlGnBu", linewidths=0.5, ax=ax, cbar_kws={'label': 'Frequency'})
            ax.set_xlabel("Year")
            ax.set_ylabel("Number")
            st.pyplot(fig_heat)

            st.subheader("Frequency Trend of Selected Numbers Over Years")
            selected_numbers = st.multiselect("Select numbers (1 to 49) for trend", options=range(1, 50), default=[7,14,23])
            if selected_numbers:
                trend_df = freq_matrix.loc[selected_numbers].transpose()
                fig_line = px.line(trend_df, x=trend_df.index, y=trend_df.columns,
                                   labels={"x": "Year", "value": "Frequency"},
                                   title="Number Frequency Trends Over Years")
                st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Predictive Model: Next Draw Number Likelihood")
        with st.spinner("Training predictive model..."):
            df_feat = build_prediction_features(numbers_df, max_draws=500)
            model, acc = train_predictive_model(df_feat)

        st.write(f"Model trained. Accuracy on test set: **{acc:.2%}**")

        probs = predict_next_draw_probs(model, numbers_df)
        probs_df = pd.DataFrame(list(probs.items()), columns=["Number", "Probability"]).sort_values(by="Probability", ascending=False)

        fig_pred = px.bar(probs_df, x="Number", y="Probability", title="Predicted Probability of Number in Next Draw", color="Probability", color_continuous_scale="Viridis")
        st.plotly_chart(fig_pred, use_container_width=True)

        budget = st.slider("Budget ($)", min_value=3, max_value=300, value=30, step=3)
        price_per_ticket = 3
        n_tickets = budget // price_per_ticket

        strategy = st.radio("Ticket generation strategy:", ["Hot/Cold mix", "Weighted by Frequency", "Advanced"])

        tickets = []
        if strategy == "Hot/Cold mix":
            tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
        elif strategy == "Weighted by Frequency":
            tickets = generate_tickets_weighted(counter, n_tickets)
        else:
            exclude_last_n = st.number_input("Exclude last N draws", min_value=0, max_value=30, value=2)
            recent_numbers = set(numbers_df.tail(exclude_last_n).values.flatten()) if exclude_last_n > 0 else set()
            fixed_numbers_input = st.text_input("Fixed numbers (comma-separated)", value="")
            fixed_numbers = set(int(x.strip()) for x in fixed_numbers_input.split(",") if x.strip().isdigit())
            due_numbers = set(gaps_df[gaps_df["Gap"] >= overdue_threshold]["Number"].tolist())
            if st.button("Generate Advanced Tickets"):
                tickets = generate_smart_tickets(n_tickets, fixed_numbers, recent_numbers, due_numbers)

        if tickets:
            st.subheader("Generated Tickets:")
            for i, t in enumerate(tickets, 1):
                st.write(f"{i}: {t}")

            csv_buffer = io.StringIO()
            pd.DataFrame(tickets, columns=[f"Num {i+1}" for i in range(6)]).to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Tickets (CSV)",
                data=csv_buffer.getvalue(),
                file_name="generated_tickets.csv",
                mime="text/csv"
            )

        st.subheader("Lottery Probability")
        st.write("""
        **P(E) = Favorable Outcomes / Total Possible Outcomes**

        For Lotto 6/49:
        - Total combinations = C(49, 6) = 13,983,816
        - Your ticket matches 1 combination.
        - **P(E)** = 1 / 13,983,816 ≈ 0.00000715%.
        """)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

else:
    st.info("Please upload a CSV file with draw numbers.")

def analyze_ticket_strategies(numbers_df, n_tickets):
    # Example: Find most frequent numbers
    counter = compute_frequencies(numbers_df)
    most_frequent_numbers = [num for num, _ in counter.most_common(6)]

    # Generate tickets
    hot_tickets = generate_tickets_hot_cold(most_frequent_numbers, n_tickets)

    # Analyze strategies
    hot_ticket_count = sum(1 for ticket in hot_tickets if set(ticket).issubset(most_frequent_numbers))

    return hot_ticket_count


