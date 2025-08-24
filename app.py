import streamlit as st
import pandas as pd
from collections import Counter
from itertools import combinations
import random
import plotly.express as px

# ======================
# Streamlit Page Config
# ======================
st.set_page_config(
    page_title="🎲 Canada Lotto 6/49 Analyzer",
    page_icon="🎲",
    layout="wide",
)
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and backtest strategies.")

# ======================
# Helper Functions
# ======================

def extract_numbers_and_bonus(df: pd.DataFrame):
    """Extract main numbers (6), optional bonus, and optional dates. Validate 1..49."""
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in main_cols):
        return None, None, None

    numbers_df = df[main_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if not numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors="coerce").dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    # Flexible date parsing
    date_col = next((col for col in ["DATE", "Draw Date", "Draw_Date", "Date"] if col in df.columns), None)
    dates = None
    if date_col:
        import re
        tmp = df[date_col].astype(str).apply(lambda x: re.sub(r"(\d+)(st|nd|rd|th)", r"\1", x))
        tmp = pd.to_datetime(tmp, errors="coerce")
        dates = tmp

    return numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df: pd.DataFrame):
    return Counter(numbers_df.values.flatten())

@st.cache_data
def compute_pair_frequencies(numbers_df: pd.DataFrame, limit: int = 500):
    counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        counts.update(combinations(sorted(row.values), 2))
    return counts

@st.cache_data
def compute_triplet_frequencies(numbers_df: pd.DataFrame, limit: int = 500):
    counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        counts.update(combinations(sorted(row.values), 3))
    return counts

def compute_number_gaps(numbers_df: pd.DataFrame, dates: pd.Series | None = None):
    last_seen = {n: -1 for n in range(1, 50)}
    df = numbers_df.copy()
    if dates is not None and len(dates) == len(numbers_df):
        df = df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    total_draws = len(df)
    return {n: (total_draws - 1 - last_seen[n]) if last_seen[n] != -1 else total_draws for n in range(1, 50)}

def most_common_per_position(numbers_df: pd.DataFrame):
    result = {}
    for col in numbers_df.columns:
        num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (int(num), int(freq))
    return result

# -------------
# Generators
# -------------

def generate_ticket(pool: list[int]):
    pool = [int(n) for n in pool]
    if len(pool) < 6:
        # Fallback to full range if pool too small
        pool = [n for n in range(1, 50)]
    return sorted(random.sample(pool, 6))

def generate_balanced_ticket():
    """3 odd / 3 even, 2 from each zone, sum in [100, 180]."""
    while True:
        ticket = []
        ticket += random.sample(range(1, 17), 2)
        ticket += random.sample(range(17, 34), 2)
        ticket += random.sample(range(34, 50), 2)
        ticket = sorted(ticket)
        odds = sum(1 for n in ticket if n % 2 == 1)
        total = sum(ticket)
        if odds == 3 and total >= 100 and total <= 180:
            return ticket

# Delta System
@st.cache_data
def compute_delta_distribution(numbers_df: pd.DataFrame):
    deltas = []
    for row in numbers_df.values:
        row = sorted(row)
        deltas.extend([row[i + 1] - row[i] for i in range(5)])
    return Counter(deltas)
    
def generate_delta_tickets(draws, n_tickets=6):
    tickets = []
    deltas = []

    # Collect deltas from historical draws
    for draw in draws:
        sorted_draw = sorted(draw)
        deltas.extend(np.diff(sorted_draw))

    # Find common deltas
    common_deltas = [d for d, _ in Counter(deltas).most_common(10)]

    for _ in range(n_tickets):
        start = random.randint(1, 10)
        seq = [start]
        for _ in range(5):
            delta = random.choice(common_deltas)
            next_num = seq[-1] + delta
            if next_num <= 49:
                seq.append(next_num)
        seq = sorted(set(seq))[:6]
        tickets.append([int(x) for x in seq])  # ✅ convert np.int64 → int

    return tickets

# Zone Coverage

def generate_zone_ticket(mode: str = "3-zone"):
    if mode == "3-zone":
        low = random.sample(range(1, 17), 2)
        mid = random.sample(range(17, 34), 2)
        high = random.sample(range(34, 50), 2)
        return sorted(low + mid + high)
    # quartiles
    q1 = random.sample(range(1, 13), 1)
    q2 = random.sample(range(13, 25), 2)
    q3 = random.sample(range(25, 37), 2)
    q4 = random.sample(range(37, 50), 1)
    return sorted(q1 + q2 + q3 + q4)

# Constraints & Exclusions

def passes_constraints(ticket: list[int], sum_min: int, sum_max: int, spread_min: int, spread_max: int, odd_count: int | None):
    total = sum(ticket)
    spread = max(ticket) - min(ticket)
    odds = sum(1 for n in ticket if n % 2 == 1)
    if odd_count is not None and odds != odd_count:
        return False
    if not (sum_min <= total <= sum_max):
        return False
    if not (spread_min <= spread <= spread_max):
        return False
    return True

def apply_exclusions_to_pool(pool: list[int], excluded: set[int]):
    pool = [n for n in pool if n not in excluded]
    if len(pool) < 6:
        # If exclusions make pool too small, relax gracefully by refilling with allowed numbers
        pool = [n for n in range(1, 50) if n not in excluded]
    return pool

# Repeat Hit
@st.cache_data
def compute_repeat_frequency(numbers_df: pd.DataFrame):
    past = [set(row) for row in numbers_df.values.tolist()]
    repeats = Counter()
    for i in range(1, len(past)):
        for n in past[i].intersection(past[i - 1]):
            repeats[n] += 1
    return repeats

def generate_repeat_ticket(last_draw: set[int], excluded: set[int], repeat_count: int = 1):
    candidates = list(last_draw - excluded)
    if len(candidates) < repeat_count:
        repeat_count = max(0, len(candidates))
    chosen_repeats = random.sample(candidates, repeat_count) if repeat_count > 0 else []
    pool = [n for n in range(1, 50) if n not in set(chosen_repeats) | excluded]
    rest = random.sample(pool, 6 - len(chosen_repeats))
    return sorted(chosen_repeats + rest)

# Simulation

def simulate_strategy(strategy_func, numbers_df: pd.DataFrame, n: int = 1000):
    past_draws = [set(row) for row in numbers_df.values.tolist()]
    results = {3: 0, 4: 0, 5: 0, 6: 0}
    for _ in range(n):
        ticket = set(strategy_func())
        for draw in past_draws:
            hits = len(ticket.intersection(draw))
            if hits >= 3:
                results[hits] += 1
    return results

# Utility – constrained ticket generation

def try_generate_with_constraints(gen_callable, *, sum_min, sum_max, spread_min, spread_max, odd_count, max_tries: int = 200):
    last_ticket = None
    for _ in range(max_tries):
        t = gen_callable()
        last_ticket = t
        if passes_constraints(t, sum_min, sum_max, spread_min, spread_max, odd_count):
            return t
    # If nothing matched constraints, return the last attempt (or None)
    return last_ticket

# ======================
# File Upload & Controls
# ======================

uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV must include columns NUMBER DRAWN 1–6. Optional: BONUS NUMBER, DATE.",
)

if not uploaded_file:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
    numbers_df, bonus_series, dates = extract_numbers_and_bonus(raw_df)
    if numbers_df is None:
        st.error("Invalid CSV. Ensure columns NUMBER DRAWN 1–6 exist and values are 1–49.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

# -------------
# Global sidebar controls
# -------------
st.sidebar.header("⚙️ Global Controls")
max_draws = len(numbers_df)
draw_limit = st.sidebar.slider("Number of past draws to analyze", min_value=10, max_value=max_draws, value=max_draws)
numbers_df = numbers_df.tail(draw_limit).reset_index(drop=True)

num_tickets = st.sidebar.slider("Tickets to generate (per tab)", 1, 12, 6)
excluded_str = st.sidebar.text_input("Exclude numbers (comma-separated)", "")
excluded = {int(x.strip()) for x in excluded_str.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 49}

sum_min, sum_max = st.sidebar.slider("Sum range", 60, 250, (100, 180))
spread_min, spread_max = st.sidebar.slider("Spread range (max - min)", 5, 48, (10, 40))
odd_mode = st.sidebar.selectbox("Odd/Even constraint", ["Any", "Exactly 0 odd", "1", "2", "3", "4", "5", "6"])  # string for nicer label
odd_count = None if odd_mode == "Any" else int(odd_mode.split()[0]) if odd_mode.startswith("Exactly") else int(odd_mode)

# =============
# Analytics
# =============

st.subheader(f"📄 Analyzed Draws: {len(numbers_df)} (from uploaded file)")
st.dataframe(numbers_df)

# Hot/Cold
counter = compute_frequencies(numbers_df)
hot = [int(n) for n, _ in counter.most_common(6)]
cold = [int(n) for n, _ in counter.most_common()[:-7:-1]]

st.markdown("### 🔥 Hot & ❄️ Cold Numbers")
st.write(f"**Hot (top 6):** {hot}")
st.write(f"**Cold (bottom 6):** {cold}")

# Most common per position
st.markdown("### 📍 Most Common Numbers by Draw Position")
pos_common = most_common_per_position(numbers_df)
cols = st.columns(6)
for i, (pos, (num, freq)) in enumerate(pos_common.items()):
    with cols[i % 6]:
        st.metric(pos, num, help=f"Appeared {freq} times")

# Frequency chart
freq_df = pd.DataFrame({"Number": list(range(1, 50)), "Frequency": [counter.get(n, 0) for n in range(1, 50)]})
fig_freq = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", color_continuous_scale="Blues", title="Number Frequency")
st.plotly_chart(fig_freq, use_container_width=True)

# Pair & Triplet charts
st.markdown("### 🤝 Pair Frequency (top 20)")
pair_counts = compute_pair_frequencies(numbers_df)
pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"]).sort_values(by="Count", ascending=False).head(20)
pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
fig_pairs = px.bar(pairs_df, x="Count", y="Pair", orientation="h", color="Count", color_continuous_scale="Viridis")
fig_pairs.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_pairs, use_container_width=True)

st.markdown("### 👪 Triplet Frequency (top 20)")
triplet_counts = compute_triplet_frequencies(numbers_df)
triplets_df = pd.DataFrame(triplet_counts.items(), columns=["Triplet", "Count"]).sort_values(by="Count", ascending=False).head(20)
triplets_df["Triplet"] = triplets_df["Triplet"].apply(lambda x: f"{x[0]} & {x[1]} & {x[2]}")
fig_triplets = px.bar(triplets_df, x="Count", y="Triplet", orientation="h", color="Count", color_continuous_scale="Cividis")
fig_triplets.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_triplets, use_container_width=True)

# Number gap analysis
st.markdown("### 🔢 Number Gap Analysis")
gaps = compute_number_gaps(numbers_df, dates=None)
gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values(by="Gap", ascending=False)
min_gap = st.slider("Show numbers with at least this many draws since last appearance:", 0, int(gaps_df["Gap"].max()), 0)
filtered_gaps = gaps_df[gaps_df["Gap"] >= min_gap].sort_values(by="Gap", ascending=False)
st.table(filtered_gaps.head(10))
fig_gap = px.bar(filtered_gaps, x="Number", y="Gap", color="Gap", color_continuous_scale="Oranges", text="Gap", title=f"Numbers with Gap ≥ {min_gap}")
fig_gap.update_traces(textposition="outside")
st.plotly_chart(fig_gap, use_container_width=True)

overdue_threshold = st.slider("Highlight numbers very overdue:", 0, int(gaps_df["Gap"].max()), int(gaps_df["Gap"].quantile(0.75)))
overdue_numbers = filtered_gaps[filtered_gaps["Gap"] >= overdue_threshold]["Number"].tolist()
st.info(f"⚠️ Very overdue numbers: {overdue_numbers}")

# Precompute for generators
predicted_numbers = [int(n) for n, _ in counter.most_common(12)]
delta_counter = compute_delta_distribution(numbers_df)
last_draw = set(numbers_df.iloc[-1])
repeats = compute_repeat_frequency(numbers_df)

# ======================
# Ticket Lab (with explanations)
# ======================

st.header("🎟️ Ticket Lab — Generators & Simulation")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Hot / Cold / Overdue",
    "Δ Delta System",
    "Cluster / Zone Coverage",
    "Sum & Spread Filters",
    "Smart Exclusion",
    "Repeat Hit Analysis",
    "Jackpot Simulation",
])

# ------- Tab 1: Hot/Cold/Overdue -------
with tab1:
    st.subheader("🔥 Hot / ❄️ Cold / ⏳ Overdue")
    st.caption(
        """
        - **Hot numbers**: most frequently drawn in the selected history window.\n
        - **Cold numbers**: least frequently drawn.\n
        - **Overdue numbers**: have not appeared for many draws (based on Gap Analysis above).\n
        Choose a bias to tilt the random generator toward these sets.
        """
    )
    choice = st.radio("Bias strategy", ["Hot", "Cold", "Overdue"], horizontal=True)

    tickets = []
    pool_base = list(range(1, 50))
    for _ in range(num_tickets):
        if choice == "Hot":
            pool = hot + [n for n in pool_base if n not in hot]
        elif choice == "Cold":
            pool = cold + [n for n in pool_base if n not in cold]
        else:
            pool = overdue_numbers + [n for n in pool_base if n not in overdue_numbers]
        pool = apply_exclusions_to_pool(pool, excluded)
        ticket = try_generate_with_constraints(
            lambda: generate_ticket(pool),
            sum_min=sum_min, sum_max=sum_max,
            spread_min=spread_min, spread_max=spread_max,
            odd_count=odd_count,
        )
        if ticket:
            tickets.append(ticket)

    st.markdown("**Generated Tickets**")
    for i, t in enumerate(tickets, 1):
        st.write(f"Ticket {i}: {t}")

# ------- Tab 2: Delta System -------
with tab2:
    st.subheader("Δ Delta System")
    st.caption(
        """
        Builds tickets from common **differences (gaps)** between consecutive numbers in historical draws.\n
        Example: sequence 4, 9, 14 → deltas +5, +5. We learn the most frequent deltas and stitch them into new sequences.
        """
    )
    tickets = []
    for _ in range(num_tickets):
        base_ticket = generate_delta_ticket(delta_counter)
        pool = apply_exclusions_to_pool(base_ticket, excluded)  # if exclusions removed values, regenerate from pool
        # If exclusions altered the base sequence, fallback to sampling from allowed pool
        gen = (lambda bt=base_ticket, p=pool: bt if len(set(bt) & excluded) == 0 else generate_ticket(p))
        ticket = try_generate_with_constraints(
            gen,
            sum_min=sum_min, sum_max=sum_max,
            spread_min=spread_min, spread_max=spread_max,
            odd_count=odd_count,
        )
        if ticket:
            tickets.append(ticket)
    st.markdown("**Generated Tickets**")
    for i, t in enumerate(tickets, 1):
        st.write(f"Ticket {i}: {t}")

# ------- Tab 3: Cluster / Zone Coverage -------
with tab3:
    st.subheader("📊 Cluster / Zone Coverage")
    st.caption(
        """
        Ensures balanced coverage across number **zones** to avoid clustering: \n
        - 3-zone: Low 1–16, Mid 17–33, High 34–49 (2 from each).\n
        - Quartiles: 1–12, 13–24, 25–36, 37–49 (1+2+2+1).
        """
    )
    mode = st.radio("Mode", ["3-zone", "Quartiles"], horizontal=True)
    tickets = []
    for _ in range(num_tickets):
        base_ticket = generate_zone_ticket("3-zone" if mode == "3-zone" else "quartiles")
        if excluded & set(base_ticket):
            # Rebuild by sampling from allowed numbers while respecting counts per zone
            if mode == "3-zone":
                low = [n for n in range(1, 17) if n not in excluded]
                mid = [n for n in range(17, 34) if n not in excluded]
                high = [n for n in range(34, 50) if n not in excluded]
                poolgen = lambda: sorted(random.sample(low, 2) + random.sample(mid, 2) + random.sample(high, 2)) if min(len(low), len(mid), len(high)) >= 2 else generate_ticket([n for n in range(1, 50) if n not in excluded])
            else:
                q1 = [n for n in range(1, 13) if n not in excluded]
                q2 = [n for n in range(13, 25) if n not in excluded]
                q3 = [n for n in range(25, 37) if n not in excluded]
                q4 = [n for n in range(37, 50) if n not in excluded]
                poolgen = lambda: sorted(random.sample(q1, 1) + random.sample(q2, 2) + random.sample(q3, 2) + random.sample(q4, 1)) if (len(q1) >= 1 and len(q2) >= 2 and len(q3) >= 2 and len(q4) >= 1) else generate_ticket([n for n in range(1, 50) if n not in excluded])
        else:
            poolgen = lambda bt=base_ticket: bt
        ticket = try_generate_with_constraints(
            poolgen,
            sum_min=sum_min, sum_max=sum_max,
            spread_min=spread_min, spread_max=spread_max,
            odd_count=odd_count,
        )
        if ticket:
            tickets.append(ticket)
    st.markdown("**Generated Tickets**")
    for i, t in enumerate(tickets, 1):
        st.write(f"Ticket {i}: {t}")

# ------- Tab 4: Sum & Spread Filters -------
with tab4:
    st.subheader("➕ Sum & Spread Filters")
    st.caption(
        """
        Filter tickets by mathematical properties: \n
        - **Sum range** (total of all 6 numbers).\n
        - **Spread** (max - min).\n
        - **Odd/Even** exact count (via sidebar).\n
        Use any generator (e.g., Balanced), then filter by your constraints.
        """
    )
    base_choice = st.selectbox("Base generator", ["Balanced", "Pure Random"]) 
    tickets = []
    for _ in range(num_tickets):
        if base_choice == "Balanced":
            gen = generate_balanced_ticket
        else:
            pool = apply_exclusions_to_pool(list(range(1, 50)), excluded)
            gen = lambda p=pool: generate_ticket(p)
        ticket = try_generate_with_constraints(
            gen,
            sum_min=sum_min, sum_max=sum_max,
            spread_min=spread_min, spread_max=spread_max,
            odd_count=odd_count,
        )
        if ticket:
            tickets.append(ticket)
    st.markdown("**Generated Tickets**")
    for i, t in enumerate(tickets, 1):
        st.write(f"Ticket {i}: {t}")

# ------- Tab 5: Smart Exclusion -------
with tab5:
    st.subheader("🚫 Smart Exclusion")
    st.caption(
        """
        Remove numbers you don't want (birthdays, unlucky, or recent hits).\n
        All generators in other tabs already respect the global **Exclude numbers** list in the sidebar.
        """
    )
    st.info(f"Currently excluding: {sorted(excluded) if excluded else 'none'}")
    pool = apply_exclusions_to_pool(list(range(1, 50)), excluded)
    tickets = []
    for _ in range(num_tickets):
        ticket = try_generate_with_constraints(
            lambda p=pool: generate_ticket(p),
            sum_min=sum_min, sum_max=sum_max,
            spread_min=spread_min, spread_max=spread_max,
            odd_count=odd_count,
        )
        if ticket:
            tickets.append(ticket)
    st.markdown("**Generated Tickets**")
    for i, t in enumerate(tickets, 1):
        st.write(f"Ticket {i}: {t}")

# ------- Tab 6: Repeat Hit Analysis -------
with tab6:
    st.subheader("🔁 Repeat Hit Analysis")
    st.caption(
        """
        Measures how often numbers **repeat** from one draw to the next.\n
        Optionally include **1–2** numbers from the **last draw** in each generated ticket.
        """
    )
    repeat_count = st.slider("How many numbers to repeat from the last draw?", 0, 2, 1)

    # Show top repeating numbers
    if repeats:
        rep_df = pd.DataFrame({"Number": list(repeats.keys()), "Consecutive Repeats": list(repeats.values())}).sort_values("Consecutive Repeats", ascending=False)
        fig_rep = px.bar(rep_df, x="Number", y="Consecutive Repeats", title="Consecutive Repeat Frequency")
        st.plotly_chart(fig_rep, use_container_width=True)

    tickets = []
    for _ in range(num_tickets):
        gen = lambda: generate_repeat_ticket(last_draw, excluded, repeat_count=repeat_count)
        ticket = try_generate_with_constraints(
            gen,
            sum_min=sum_min, sum_max=sum_max,
            spread_min=spread_min, spread_max=spread_max,
            odd_count=odd_count,
        )
        if ticket:
            tickets.append(ticket)
    st.markdown("**Generated Tickets**")
    for i, t in enumerate(tickets, 1):
        st.write(f"Ticket {i}: {t}")

# ------- Tab 7: Jackpot Simulation -------
with tab7:
    st.subheader("🎰 Jackpot Pattern Simulation")
    st.caption(
        """
        Simulate many tickets using a chosen generator and compare against **historical draws**.\n
        We count how often the tickets would have matched **3/6, 4/6, 5/6, 6/6** numbers.
        """
    )
    sim_n = st.slider("Number of tickets to simulate", 200, 5000, 1000, step=100)
    sim_strategy = st.selectbox("Strategy to simulate", [
        "Balanced", "Pure Random", "Hot Bias", "Delta System", "Zone (3-zone)", "Zone (Quartiles)", "Repeat (1)", "Repeat (2)",
    ])

    def make_strategy_callable(name: str):
        if name == "Balanced":
            return generate_balanced_ticket
        if name == "Pure Random":
            pool = apply_exclusions_to_pool(list(range(1, 50)), excluded)
            return (lambda p=pool: generate_ticket(p))
        if name == "Hot Bias":
            pool = apply_exclusions_to_pool(hot + [n for n in range(1, 50) if n not in hot], excluded)
            return (lambda p=pool: generate_ticket(p))
        if name == "Delta System":
            return (lambda: generate_delta_ticket(delta_counter))
        if name == "Zone (3-zone)":
            return (lambda: generate_zone_ticket("3-zone"))
        if name == "Zone (Quartiles)":
            return (lambda: generate_zone_ticket("quartiles"))
        if name == "Repeat (1)":
            return (lambda: generate_repeat_ticket(last_draw, excluded, repeat_count=1))
        if name == "Repeat (2)":
            return (lambda: generate_repeat_ticket(last_draw, excluded, repeat_count=2))
        return (lambda: generate_ticket(apply_exclusions_to_pool(list(range(1, 50)), excluded)))

    if st.button("Run Simulation"):
        func = make_strategy_callable(sim_strategy)
        results = simulate_strategy(func, numbers_df, n=sim_n)
        st.json(results)
        # Simple bar chart of results
        res_df = pd.DataFrame({"Matches": list(results.keys()), "Count": list(results.values())}).sort_values("Matches")
        fig_res = px.bar(res_df, x="Matches", y="Count", title="Simulation Results")
        st.plotly_chart(fig_res, use_container_width=True)

# ======================
# Notes
# ======================

st.markdown(
    """
    **Disclaimer:** Lotto 6/49 is a game of chance. These analyses and generators cannot predict future draws, 
    but they can help you explore patterns, enforce constraints, and compare strategies on historical data.
    """
)
