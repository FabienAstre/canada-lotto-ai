import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations
import random

st.set_page_config(page_title="Lotto 6/49 Honest Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Lotto 6/49 — Honest Analyzer")
st.caption(
    "No ML. No snake oil. Just real statistics, split-avoidance logic, and honest generators. "
    "Every combo has 1 in 13,983,816 odds — this tool helps you pick smarter, not luckier."
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

NUMBER_COLS = [f"NUMBER DRAWN {i}" for i in range(1, 7)]

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if not all(c in df.columns for c in NUMBER_COLS):
        return None
    df[NUMBER_COLS] = df[NUMBER_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=NUMBER_COLS)
    df[NUMBER_COLS] = df[NUMBER_COLS].astype(int)
    valid = df[NUMBER_COLS].apply(lambda r: r.between(1, 49).all(), axis=1)
    df = df[valid].reset_index(drop=True)
    bonus_col = next((c for c in df.columns if "bonus" in c.lower()), None)
    if bonus_col:
        df["BONUS"] = pd.to_numeric(df[bonus_col], errors="coerce").astype("Int64")
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col and date_col != "DATE":
        df["DATE"] = pd.to_datetime(df[date_col], errors="coerce")
    return df

@st.cache_data
def frequency_table(df: pd.DataFrame) -> dict:
    return Counter(df[NUMBER_COLS].values.flatten().tolist())

@st.cache_data
def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    last_seen = {}
    total = len(df)
    for idx, row in df.iterrows():
        for n in row[NUMBER_COLS]:
            last_seen[n] = idx
    rows = []
    for n in range(1, 50):
        last = last_seen.get(n, -1)
        gap = (total - 1 - last) if last >= 0 else total
        rows.append({"Number": n, "Gap (draws since last seen)": gap})
    return pd.DataFrame(rows).sort_values("Gap (draws since last seen)", ascending=False).reset_index(drop=True)

@st.cache_data
def pair_freq(df: pd.DataFrame) -> pd.DataFrame:
    c = Counter()
    for _, row in df.iterrows():
        c.update(combinations(sorted(row[NUMBER_COLS].tolist()), 2))
    top = c.most_common(25)
    return pd.DataFrame([{"Pair": f"{a} & {b}", "Count": cnt} for (a, b), cnt in top])

@st.cache_data
def decade_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    bands = {"1–9": range(1, 10), "10–19": range(10, 20), "20–29": range(20, 30),
             "30–39": range(30, 40), "40–49": range(40, 50)}
    rows = []
    for label, rng in bands.items():
        s = set(rng)
        total = sum(1 for n in df[NUMBER_COLS].values.flatten() if n in s)
        avg = total / len(df)
        rows.append({"Decade": label, "Total appearances": total, "Avg per draw": round(avg, 2)})
    return pd.DataFrame(rows)

# popularity score: lower = fewer players likely pick it → less splitting
def popularity_score(n: int) -> float:
    # Birthday penalty (1–31 heavily played), teen numbers extra popular
    score = 1.0
    if n <= 12:   score += 0.8   # months + very low numbers
    elif n <= 31: score += 0.4   # birthday zone
    else:         score -= 0.3   # above 31 = less popular
    if n in [7, 3, 11, 13, 17, 21]:  score += 0.3  # "lucky" numbers
    return round(score, 2)

def split_risk(ticket: list) -> str:
    avg = sum(popularity_score(n) for n in ticket) / 6
    if avg > 1.2:  return "🔴 High split risk — too many popular/birthday numbers"
    if avg > 0.9:  return "🟡 Medium split risk"
    return "🟢 Low split risk — good unpopular territory"

def has_consec(t): 
    s = sorted(t)
    return any(s[i+1] - s[i] == 1 for i in range(5))

def is_arith(t):
    s = sorted(t)
    gaps = [s[i+1] - s[i] for i in range(5)]
    return len(set(gaps)) == 1

def passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
    if not (sum_min <= sum(t) <= sum_max): return False
    if sum(1 for n in t if n > 31) < above31_min: return False
    if no_consec and has_consec(t): return False
    if no_arith and is_arith(t): return False
    odds = sum(1 for n in t if n % 2 != 0)
    if not (odd_min <= odds <= odd_max): return False
    return True

def gen_ticket(pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=5000):
    p = [n for n in pool if 1 <= n <= 49]
    if len(p) < 6:
        p = list(range(1, 50))
    for _ in range(tries):
        t = sorted(random.sample(p, 6))
        if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            return t
    return sorted(random.sample(p, 6))  # fallback, no constraint

def decade_spread_ticket(excluded, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=10000):
    d1 = [n for n in range(1, 10)  if n not in excluded]
    d2 = [n for n in range(10, 20) if n not in excluded]
    d3 = [n for n in range(20, 30) if n not in excluded]
    hi = [n for n in range(30, 50) if n not in excluded]
    for _ in range(tries):
        if not d1 or not d2 or not d3 or len(hi) < 3: break
        t = sorted([random.choice(d1), random.choice(d2), random.choice(d3)] + random.sample(hi, 3))
        if len(set(t)) < 6: continue
        if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            return t
    return None

def delta_ticket(delta_dist, excluded, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=5000):
    top_deltas = [d for d, _ in delta_dist.most_common(15)] or list(range(1, 10))
    pool = [n for n in range(1, 50) if n not in excluded]
    for _ in range(tries):
        start = random.randint(1, 15)
        seq = [start]
        for _ in range(5):
            seq.append(seq[-1] + random.choice(top_deltas))
        seq = [n for n in seq if 1 <= n <= 49 and n not in excluded]
        if len(set(seq)) == 6:
            t = sorted(seq)
            if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
                return t
    return gen_ticket(pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload Lotto 6/49 CSV", type=["csv"],
    help="Must have columns: NUMBER DRAWN 1–6. Optional: BONUS NUMBER, DATE.")

sum_min = st.sidebar.slider("Min sum", 60, 180, 115)
sum_max = st.sidebar.slider("Max sum", 120, 250, 185)
above31_min = st.sidebar.slider("Min numbers above 31 (split protection)", 0, 6, 3)
odd_min, odd_max = st.sidebar.select_slider("Odd count range", options=list(range(7)), value=(2, 4))
no_consec = st.sidebar.checkbox("No consecutive numbers", value=True)
no_arith = st.sidebar.checkbox("No arithmetic sequences", value=True)
n_tickets = st.sidebar.slider("Tickets to generate", 1, 10, 4)
excl_str = st.sidebar.text_input("Exclude numbers (comma-separated)", "")
excluded = {int(x.strip()) for x in excl_str.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 49}
if excluded:
    st.sidebar.info(f"Excluding: {sorted(excluded)}")

# ──────────────────────────────────────────────
# LUCKY COMBO CHECK
# ──────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("🩷 Your lucky combo")
lucky_str = st.sidebar.text_input("Enter your always-play numbers", "3,9,10,12,17,25")

# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────

if not uploaded:
    st.info("👆 Upload a CSV in the sidebar to unlock all analytics. Generators work without a file too.")

    # ── Generators without data ──
    st.header("🎟️ Ticket Generators — no data needed")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decade spread")
        st.caption("1 from each of 1–9, 10–19, 20–29, then 3 from 30–49")
        if st.button("Generate (decade spread)", key="g1"):
            tickets = []
            for _ in range(n_tickets):
                t = decade_spread_ticket(excluded, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
                if t: tickets.append(t)
            for i, t in enumerate(tickets, 1):
                risk = split_risk(t)
                st.markdown(f"**Draw {i}:** `{t}` — Sum {sum(t)} — {risk}")

    with col2:
        st.subheader("Pure filtered random")
        st.caption("Fully random but must pass all your sidebar constraints")
        if st.button("Generate (filtered random)", key="g2"):
            pool = [n for n in range(1, 50) if n not in excluded]
            tickets = [gen_ticket(pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max) for _ in range(n_tickets)]
            for i, t in enumerate(tickets, 1):
                risk = split_risk(t)
                st.markdown(f"**Draw {i}:** `{t}` — Sum {sum(t)} — {risk}")

    # ── Lucky combo audit ──
    if lucky_str.strip():
        st.markdown("---")
        st.subheader("🩷 Lucky combo audit")
        try:
            lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
            if len(lucky) == 6:
                s = sum(lucky)
                above = sum(1 for n in lucky if n > 31)
                odds = sum(1 for n in lucky if n % 2 != 0)
                risk = split_risk(lucky)
                consec = has_consec(lucky)
                arith = is_arith(lucky)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sum", s, delta=f"{'OK' if 115<=s<=185 else 'OUT OF RANGE'}")
                col2.metric("Above 31", above, delta=f"{'OK' if above>=3 else 'LOW'}")
                col3.metric("Odd count", odds)
                col4.metric("Consecutive?", "Yes" if consec else "No")

                st.write(f"**Numbers:** {lucky}")
                st.write(f"**Split risk:** {risk}")
                if s < 115:
                    st.error(f"Sum {s} is below the 115–185 historical range — all numbers cluster too low.")
                if above == 0:
                    st.error("Zero numbers above 31 — entirely in birthday territory. Very high split risk.")
                if consec:
                    st.warning("Contains consecutive numbers.")
        except:
            st.warning("Enter 6 comma-separated numbers.")
    st.stop()

# ──────────────────────────────────────────────
# WITH DATA
# ──────────────────────────────────────────────

df = load_csv(uploaded)
if df is None:
    st.error("CSV must contain columns: NUMBER DRAWN 1–6 with values 1–49.")
    st.stop()

draw_limit = st.slider("Draws to analyze", 10, len(df), len(df))
df = df.tail(draw_limit).reset_index(drop=True)

st.success(f"✅ {len(df)} draws loaded.")

tab_stats, tab_gaps, tab_decades, tab_pairs, tab_gen, tab_lucky, tab_check = st.tabs([
    "📊 Frequency", "⏳ Gap Analysis", "🔢 Decade Breakdown",
    "🤝 Pairs", "🎟️ Generators", "🩷 Lucky Combo", "🔍 Check a Combo"
])

# ── Tab: Frequency ──
with tab_stats:
    st.subheader("Number frequency — all time")
    freq = frequency_table(df)
    freq_df = pd.DataFrame({"Number": list(range(1, 50)), "Frequency": [freq.get(n, 0) for n in range(1, 50)]})
    fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                 color_continuous_scale="Blues", title="How often each number has appeared")
    st.plotly_chart(fig, use_container_width=True)

    hot6 = [n for n, _ in freq.most_common(6)]
    cold6 = [n for n, _ in freq.most_common()[:-7:-1]]
    c1, c2 = st.columns(2)
    c1.metric("🔥 Top 6 hot", str(hot6))
    c2.metric("❄️ Bottom 6 cold", str(cold6))

# ── Tab: Gap Analysis ──
with tab_gaps:
    st.subheader("Number gap analysis — draws since last appearance")
    gaps_df = gap_table(df)
    fig2 = px.bar(gaps_df, x="Number", y="Gap (draws since last seen)",
                  color="Gap (draws since last seen)", color_continuous_scale="Oranges",
                  title="Draws since each number last appeared")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Most overdue numbers:**")
    st.dataframe(gaps_df.head(10), use_container_width=True)

    overdue_threshold = int(gaps_df["Gap (draws since last seen)"].quantile(0.75))
    overdue = gaps_df[gaps_df["Gap (draws since last seen)"] >= overdue_threshold]["Number"].tolist()
    st.info(f"⚠️ Numbers overdue (top 25%): {overdue}")
    st.caption("Reminder: overdue numbers are not statistically more likely — this is for exploration only.")

# ── Tab: Decade breakdown ──
with tab_decades:
    st.subheader("How many numbers per decade range appear in a typical draw")
    dec_df = decade_breakdown(df)
    fig3 = px.bar(dec_df, x="Decade", y="Avg per draw", color="Avg per draw",
                  color_continuous_scale="Teal", text="Avg per draw",
                  title="Average numbers drawn per decade range")
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(dec_df, use_container_width=True)
    st.caption("The 40–49 range slightly dominates historically. The 1–9 range has one fewer ball (9 vs 10) so naturally appears less.")

# ── Tab: Pairs ──
with tab_pairs:
    st.subheader("Most common pairs")
    p_df = pair_freq(df)
    fig4 = px.bar(p_df, x="Count", y="Pair", orientation="h",
                  color="Count", color_continuous_scale="Viridis",
                  title="Top 25 most co-occurring pairs")
    fig4.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig4, use_container_width=True)

# ── Tab: Generators ──
with tab_gen:
    st.subheader("🎟️ Ticket generators")

    freq = frequency_table(df)
    gaps_df = gap_table(df)
    hot_pool = [n for n, _ in freq.most_common(20) if n not in excluded]
    cold_pool = [n for n, _ in freq.most_common()[:-21:-1] if n not in excluded]
    overdue_pool = gaps_df.head(15)["Number"].tolist()
    overdue_pool = [n for n in overdue_pool if n not in excluded]
    full_pool = [n for n in range(1, 50) if n not in excluded]

    last_draw = set(df.iloc[-1][NUMBER_COLS].tolist())
    recent_drawn = set()
    for _, row in df.tail(3).iterrows():
        recent_drawn.update(row[NUMBER_COLS].tolist())
    fresh_pool = [n for n in range(1, 50) if n not in excluded and n not in recent_drawn]

    delta_dist = Counter()
    for _, row in df.iterrows():
        s = sorted(row[NUMBER_COLS].tolist())
        delta_dist.update(s[i+1] - s[i] for i in range(5))

    gen_mode = st.selectbox("Generator strategy", [
        "Decade spread (recommended)",
        "Filtered random",
        "Hot numbers biased",
        "Cold numbers biased",
        "Overdue numbers biased",
        "Avoids last 3 draws",
        "Delta system",
    ])

    if st.button("🎲 Generate tickets"):
        tickets = []
        for _ in range(n_tickets):
            if gen_mode == "Decade spread (recommended)":
                t = decade_spread_ticket(excluded, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Filtered random":
                t = gen_ticket(full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Hot numbers biased":
                t = gen_ticket(hot_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Cold numbers biased":
                t = gen_ticket(cold_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Overdue numbers biased":
                t = gen_ticket(overdue_pool + full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Avoids last 3 draws":
                t = gen_ticket(fresh_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Delta system":
                t = delta_ticket(delta_dist, excluded, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            else:
                t = gen_ticket(full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            if t:
                tickets.append(t)

        st.markdown("---")
        for i, t in enumerate(tickets, 1):
            risk = split_risk(t)
            above = sum(1 for n in t if n > 31)
            s = sum(t)
            odds = sum(1 for n in t if n % 2 != 0)
            st.markdown(
                f"**Draw {i}:** `{t}`  \n"
                f"Sum: **{s}** | Above 31: **{above}** | Odd/Even: **{odds}/{6-odds}** | {risk}"
            )

# ── Tab: Lucky Combo ──
with tab_lucky:
    st.subheader("🩷 Lucky combo audit")
    if lucky_str.strip():
        try:
            lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
            if len(lucky) == 6:
                s = sum(lucky)
                above = sum(1 for n in lucky if n > 31)
                odds = sum(1 for n in lucky if n % 2 != 0)
                risk = split_risk(lucky)
                consec = has_consec(lucky)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sum", s, delta="OK" if 115 <= s <= 185 else "⚠️ Out of range")
                col2.metric("Above 31", f"{above}/6", delta="OK" if above >= 3 else "⚠️ Too low")
                col3.metric("Odd / Even", f"{odds} / {6-odds}")
                col4.metric("Consecutive?", "Yes ⚠️" if consec else "No ✓")

                st.write(f"**Split risk:** {risk}")

                freq = frequency_table(df)
                gaps_df = gap_table(df)
                lfreq = [(n, freq.get(n, 0)) for n in lucky]
                lgap = [(n, gaps_df.set_index("Number").loc[n, "Gap (draws since last seen)"]) for n in lucky]

                st.markdown("**Your numbers in context:**")
                ctx_df = pd.DataFrame({
                    "Number": lucky,
                    "All-time frequency": [f[1] for f in lfreq],
                    "Draws since last seen": [g[1] for g in lgap],
                    "Popularity score": [popularity_score(n) for n in lucky],
                    "Above 31?": ["Yes" if n > 31 else "No" for n in lucky],
                })
                st.dataframe(ctx_df, use_container_width=True)

                if s < 115:
                    st.error(f"Sum {s} is well below the 115–185 historical range.")
                if above == 0:
                    st.error("All numbers are in the birthday zone (≤31). Very high jackpot-split risk.")
                if consec:
                    st.warning("Contains at least one pair of consecutive numbers.")
        except Exception as e:
            st.warning(f"Could not parse lucky numbers: {e}")

# ── Tab: Check a combo ──
with tab_check:
    st.subheader("🔍 Has this combination ever been drawn?")
    check_str = st.text_input("Enter 6 numbers (comma-separated):", "")
    if check_str.strip():
        try:
            check = tuple(sorted(int(x.strip()) for x in check_str.split(",") if x.strip().isdigit()))
            if len(check) != 6:
                st.warning("Enter exactly 6 numbers.")
            else:
                past = [tuple(sorted(row[NUMBER_COLS].tolist())) for _, row in df.iterrows()]
                count = past.count(check)
                if count > 0:
                    st.success(f"✅ This combination appeared {count} time(s) in the {len(df)} draws analyzed.")
                else:
                    st.info(f"❌ Never appeared in the {len(df)} draws analyzed.")
        except:
            st.warning("Invalid input.")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────

st.markdown("---")
st.caption(
    "**The honest truth:** Lotto 6/49 is a certified random draw. No tool, model, or strategy changes your odds "
    "(1 in 13,983,816 per ticket). The only real edge is split avoidance — picking unpopular numbers above 31 "
    "means if you win, you're less likely to share. Everything else is pattern exploration for fun."
)
