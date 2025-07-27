import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np
import io

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze real draws, statistics, and generate smart tickets.")

# --- Helpers ---
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

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {num: -1 for num in range(1, 50)}
    gaps = {num: None for num in range(1, 50)}

    if dates is not None:
        order = dates.argsort()
        numbers_df = numbers_df.iloc[order].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for num in range(1, 50):
            if last_seen[num] == -1:
                gaps[num] = idx
            else:
                gaps[num] = idx - last_seen[num]
        for n in row.values:
            last_seen[n] = idx

    total_draws = len(numbers_df)
    for num in range(1, 50):
        if last_seen[num] != -1:
            gaps[num] = total_draws - 1 - last_seen[num]
        else:
            gaps[num] = total_draws

    return gaps

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to NUMBER DRAWN 6 and BONUS NUMBER",
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data:")
        st.dataframe(df)

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("The CSV file must contain columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with valid numbers between 1 and 49.")
        else:
            st.subheader("Latest Draws:")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Bonus Numbers (latest draws):")
                st.write(bonus_series.tail(30).to_list())

            # --- Frequency Analysis ---
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
                title="Number Frequency (all draws)",
                labels={"Number": "Number", "Frequency": "Occurrences"},
                color="Frequency",
                color_continuous_scale="Blues",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # --- Hot vs Cold Chart ---
            hot_df = freq_df[freq_df["Number"].isin(hot)]
            cold_df = freq_df[freq_df["Number"].isin(cold)]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=hot_df["Number"], y=hot_df["Frequency"], name="Hot Numbers", marker_color="red"))
            fig2.add_trace(go.Bar(x=cold_df["Number"], y=cold_df["Frequency"], name="Cold Numbers", marker_color="blue"))
            fig2.update_layout(
                barmode="group",
                title="Hot vs Cold Numbers",
                xaxis_title="Number",
                yaxis_title="Frequency",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # --- Pair Frequency Chart (No Table) ---
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

            # --- Improved Gap Analysis ---
            st.subheader("Number Gap Analysis (Improved)")
            gaps = compute_number_gaps(numbers_df, dates)
            gap_list = [{"Number": num, "Gap": gaps[num]} for num in range(1, 50)]
            gaps_df = pd.DataFrame(gap_list).sort_values(by="Gap", ascending=False)

            avg_gap = np.mean(list(gaps.values()))
            median_gap = np.median(list(gaps.values()))
            st.write(f"**Average gap:** {avg_gap:.2f} draws")
            st.write(f"**Median gap:** {median_gap:.2f} draws")

            overdue_threshold = st.slider(
                "Gap threshold for overdue numbers (draws)", min_value=0, max_value=100, value=27
            )
            overdue_df = gaps_df[gaps_df["Gap"] >= overdue_threshold]

            st.write(f"**Overdue numbers (gap ≥ {overdue_threshold} draws):**")
            st.dataframe(overdue_df)

            top_overdue = gaps_df.head(10)
            fig_gaps = px.bar(
                top_overdue,
                x="Gap",
                y="Number",
                orientation='h',
                title="Top 10 Overdue Numbers",
                color="Gap",
                color_continuous_scale="Reds"
            )
            fig_gaps.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_gaps, use_container_width=True)

            if not overdue_df.empty:
                recommended_nums = overdue_df.head(5)["Number"].tolist()
                st.info(f"**Recommendation:** The top 5 overdue numbers are: {', '.join(map(str, recommended_nums))}.")
            else:
                st.info("No numbers exceed the selected overdue threshold.")

            # --- Ticket Generation ---
            budget = st.slider("Budget ($)", min_value=3, max_value=300, value=30, step=3)
            price_per_ticket = 3
            n_tickets = budget // price_per_ticket

            strategy = st.radio("Choose ticket generation strategy:",
                                ("Hot/Cold mix (original)", "Weighted by Frequency (new)", "Advanced (fixed, exclusions, overdue)"))

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

            tickets = []
            if strategy == "Advanced (fixed, exclusions, overdue)":
                exclude_last_n = st.number_input("Exclude numbers drawn in last N draws", min_value=0, max_value=30, value=2)
                recent_numbers = set(numbers_df.tail(exclude_last_n).values.flatten()) if exclude_last_n > 0 else set()

                include_due = st.checkbox("Include overdue numbers (gap analysis)", value=True)
                due_numbers = set(overdue_df["Number"].tolist()) if include_due and not overdue_df.empty else set()

                fixed_numbers_input = st.text_input("Enter your fixed numbers (comma-separated)", value="")
                fixed_numbers = set()
                if fixed_numbers_input.strip():
                    try:
                        fixed_numbers = set(int(x.strip()) for x in fixed_numbers_input.split(",") if 1 <= int(x.strip()) <= 49)
                    except:
                        st.error("Please enter valid numbers between 1 and 49, separated by commas.")
                if len(fixed_numbers) > 5:
                    st.error("You can fix a maximum of 5 numbers.")
                    fixed_numbers = set(list(fixed_numbers)[:5])

                if st.button("Generate Advanced Tickets"):
                    tickets = generate_smart_tickets(
                        n_tickets=n_tickets,
                        fixed_nums=fixed_numbers,
                        exclude_nums=recent_numbers,
                        due_nums=due_numbers
                    )
            else:
                if strategy == "Hot/Cold mix (original)":
                    tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
                else:
                    tickets = generate_tickets_weighted(counter, n_tickets)

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

            # --- Probability Explanation ---
            st.subheader("Understanding Lottery Probability")
            st.write("""
            **P(E) = Favorable Outcomes / Total Possible Outcomes**

            - **P(E)**: Probability of the event (E) happening.
            - **Favorable Outcomes**: Number of ways the specific event can occur.
            - **Total Possible Outcomes**: Total number of all possible results.
            """)
            st.write("""
            **Example (6/49 Lottery):**

            - We choose 6 numbers from 49.
            - **Total Possible Outcomes** = C(49, 6) = 13,983,816.
            - **Favorable Outcomes** = 1 (only your exact 6-number combination).
            - Therefore, **P(E)** = 1 / 13,983,816 ≈ 0.0000000715 ≈ 0.00000715%.

            This is why winning the jackpot is extremely rare.
            """)

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")

else:
    st.info("Please upload a CSV file with the draw numbers.")
