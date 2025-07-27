import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
def build_prediction_features(numbers_df, max_draws=100):
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
def train_predictive_model(df_feat, model_type="random_forest"):
    feature_cols = ['gap', 'frequency']
    X = df_feat[feature_cols]
    y = df_feat['appeared_next']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")

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

def get_top_ticket_from_probs(probs_dict):
    sorted_nums = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    top6 = tuple(num for num, prob in sorted_nums[:6])
    return top6

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

        st.subheader("Yearly Number Frequency Heatmap")
        if dates is not None:
            numbers_df_with_dates = numbers_df.copy()
            numbers_df_with_dates['Year'] = dates.dt.year.values[:len(numbers_df)]
            years = sorted(numbers_df_with_dates['Year'].unique())
            selected_years = st.multiselect("Select years to visualize", years, default=years[-5:])
            if selected_years:
                subset = numbers_df_with_dates[numbers_df_with_dates['Year'].isin(selected_years)]
                freq_by_year = pd.DataFrame()
                for y in selected_years:
                    draws = subset[subset['Year'] == y]
                    counts = Counter(draws.values.flatten())
                    freq_by_year[y] = pd.Series({k: counts.get(k, 0) for k in range(1, 50)})
                freq_by_year = freq_by_year.fillna(0)
                plt.figure(figsize=(12,6))
                sns.heatmap(freq_by_year.T, cmap="YlGnBu", cbar=True)
                st.pyplot(plt.gcf())

        st.subheader("Ticket Generation")
        budget = st.slider("Budget in $", min_value=3, max_value=300, value=30, step=3)
        price_per_ticket = 3
        n_tickets = budget // price_per_ticket

        gen_method = st.selectbox("Ticket generation method", ["Hot/Cold Weighted", "Frequency Weighted", "Smart Generation"])

        if gen_method == "Hot/Cold Weighted":
            tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
        elif gen_method == "Frequency Weighted":
            tickets = generate_tickets_weighted(counter, n_tickets)
        else:
            fixed_nums = st.multiselect("Fix numbers to include in tickets (optional)", options=list(range(1, 50)))
            exclude_nums = set()
            if st.checkbox("Exclude numbers drawn in last 2 draws?"):
                exclude_nums = set(numbers_df.tail(2).values.flatten())
            overdue_nums = set([num for num, gap in gaps.items() if gap >= overdue_threshold])
            tickets = generate_smart_tickets(n_tickets, set(fixed_nums), exclude_nums, overdue_nums)

        st.write(f"Generated Tickets ({len(tickets)}):")
        for i, t in enumerate(tickets, 1):
            st.write(f"{i}: {t}")

        st.subheader("Predictive Models for Next Draw Number Likelihood and Suggested Tickets")

        models = {
            "Logistic Regression": "logistic",
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting"
        }

        tickets_by_model = {}

        for model_name, model_type in models.items():
            with st.spinner(f"Training {model_name} model on last 100 draws..."):
                df_feat = build_prediction_features(numbers_df, max_draws=100)
                model, acc = train_predictive_model(df_feat, model_type=model_type)

            st.write(f"**{model_name}** trained. Accuracy on test set: **{acc:.2%}**")

            probs = predict_next_draw_probs(model, numbers_df)
            probs_df = pd.DataFrame(list(probs.items()), columns=["Number", "Probability"]).sort_values(by="Probability", ascending=False)

            fig_pred = px.bar(probs_df, x="Number", y="Probability",
                              title=f"Predicted Probability of Number in Next Draw ({model_name})",
                              color="Probability", color_continuous_scale="Viridis")
            st.plotly_chart(fig_pred, use_container_width=True)

            # Generate suggested ticket (top 6 numbers by probability)
            ticket = get_top_ticket_from_probs(probs)
            tickets_by_model[model_name] = ticket

        st.subheader("Suggested Tickets from Each Predictive Model")
        for model_name, ticket in tickets_by_model.items():
            st.write(f"{model_name}: {ticket}")

    except Exception as e:
        st.error(f"Error processing CSV: {e}")

else:
    st.info("Please upload a CSV file with Lotto 6/49 draws.")
