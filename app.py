import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze real draws, statistics and generate tickets including bonus number.")

uploaded_file = st.file_uploader(
    "Import a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV must include columns: NUMBER DRAWN 1 to NUMBER DRAWN 6 and BONUS NUMBER",
)

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

def generate_tickets_hot_cold(hot, cold, n_tickets):
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

        # Pick bonus number NOT in main numbers
        bonus_pool = set(range(1, pool+1)) - set(main_numbers)
        bonus_number = random.choice(list(bonus_pool))

        ticket = main_numbers + (bonus_number,)
        tickets.add(ticket)

    return list(tickets)

def generate_tickets_weighted(counter, n_tickets):
    tickets = set()
    pool = 49
    main_needed = 6
    numbers = list(range(1, pool + 1))
    weights = [counter.get(num, 1) for num in numbers]  # frequency as weights

    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    while len(tickets) < n_tickets:
        main_numbers = set()
        while len(main_numbers) < main_needed:
            main_numbers.add(random.choices(numbers, probs)[0])

        main_numbers = tuple(sorted(main_numbers))

        bonus_pool = set(numbers) - set(main_numbers)
        bonus_number = random.choice(list(bonus_pool))

        ticket = main_numbers + (bonus_number,)
        tickets.add(ticket)

    return list(tickets)

def generate_smart_tickets(n_tickets, fixed_nums, exclude_nums, overdue_nums):
    tickets = set()
    pool = 49
    main_needed = 6

    while len(tickets) < n_tickets:
        main_numbers = set(fixed_nums)

        # Add overdue numbers if not excluded and space available
        for num in overdue_nums:
            if num not in exclude_nums and len(main_numbers) < main_needed:
                main_numbers.add(num)

        available_nums = set(range(1, pool + 1)) - main_numbers - exclude_nums
        while len(main_numbers) < main_needed:
            main_numbers.add(random.choice(list(available_nums)))

        main_numbers = tuple(sorted(main_numbers))

        bonus_pool = set(range(1, pool + 1)) - set(main_numbers)
        bonus_number = random.choice(list(bonus_pool))

        ticket = main_numbers + (bonus_number,)
        tickets.add(ticket)

    return list(tickets)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Imported Data:")
        st.dataframe(df)

        numbers_df, bonus_series = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("CSV must contain valid 6 main number columns between 1 and 49.")
        else:
            st.subheader("Last 30 Draws (Main Numbers):")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Last 30 Bonus Numbers:")
                st.write(bonus_series.tail(30).to_list())

            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)

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

            freq_df = pd.DataFrame({"Number": list(range(1, 50))})
            freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter[x] if x in counter else 0)

            fig = px.bar(
                freq_df,
                x="Number",
                y="Frequency",
                title="Frequency of Numbers (All Imported Draws)",
                labels={"Number": "Number", "Frequency": "Occurrences"},
                color="Frequency",
                color_continuous_scale="Blues",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            hot_df = freq_df[freq_df["Number"].isin(hot)]
            cold_df = freq_df[freq_df["Number"].isin(cold)]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=hot_df["Number"], y=hot_df["Frequency"], name="Hot Numbers", marker_color="red"))
            fig2.add_trace(go.Bar(x=cold_df["Number"], y=cold_df["Frequency"], name="Cold Numbers", marker_color="blue"))
            fig2.update_layout(
                barmode="group",
                title="Hot vs Cold Numbers Comparison",
                xaxis_title="Number",
                yaxis_title="Frequency",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

            pair_counts = Counter()
            for _, row in numbers_df.iterrows():
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
                title="Number Pair Frequency",
                labels={"Count": "Occurrences", "Pair": "Number Pair"},
                color="Count",
                color_continuous_scale="Viridis",
            )
            fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_pairs, use_container_width=True)

            budget = st.slider("Budget in $", min_value=3, max_value=300, value=30, step=3)
            price_per_ticket = 3
            n_tickets = budget // price_per_ticket

            gen_method = st.selectbox("Ticket generation method", ["Hot/Cold Weighted", "Frequency Weighted", "Smart Generation"])

            # Generate tickets based on selection
            if gen_method == "Hot/Cold Weighted":
                tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
            elif gen_method == "Frequency Weighted":
                tickets = generate_tickets_weighted(counter, n_tickets)
            else:
                fixed_nums = st.multiselect("Fix numbers to include in tickets (optional)", options=list(range(1, 50)))
                exclude_nums = set()
                if st.checkbox("Exclude numbers drawn in last 2 draws?"):
                    exclude_nums = set(numbers_df.tail(2).values.flatten())
                overdue_nums = {}  # You can add overdue logic here
                tickets = generate_smart_tickets(n_tickets, set(fixed_nums), exclude_nums, set(overdue_nums.keys()))

            st.subheader(f"Generated Tickets ({len(tickets)}):")
            for i, t in enumerate(tickets, 1):
                main_nums = t[:-1]
                bonus_num = t[-1]
                st.write(f"{i}: Main Numbers: {main_nums} | Bonus Number: {bonus_num}")

            # Predictive models section is independent and always visible
            st.subheader("Predictive Models for Next Draw Number Likelihood")

            # Example placeholder predictive model names (you can implement actual models)
            predictive_models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            selected_models = st.multiselect("Select predictive models to run", predictive_models, default=predictive_models)

            for model_name in selected_models:
                st.write(f"Running {model_name} model... (placeholder)")
                # Implement model training and prediction here
                # Display prediction results as needed

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

else:
    st.info("Please upload a CSV file with Lotto 6/49 draw numbers.")
