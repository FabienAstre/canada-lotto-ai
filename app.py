import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from collections import Counter
from itertools import combinations
import random
import re
import time
from datetime import datetime

st.set_page_config(page_title="Lotto 6/49 Live Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Lotto 6/49 — Live Analyzer")
st.caption(
    "Pulls real draw results directly from the web. No CSV. No ML. No snake oil. "
    "Just honest statistics and split-avoidance logic. Every combo = 1 in 13,983,816 odds."
)

# ──────────────────────────────────────────────
# LIVE DATA FETCHING
# ──────────────────────────────────────────────

MONTHS = {
    "Jan": "0126", "Feb": "0226", "Mar": "0326", "Apr": "0426",
    "May": "0526", "Jun": "0626", "Jul": "0726", "Aug": "0826",
    "Sep": "0926", "Oct": "1026", "Nov": "1126", "Dec": "1226",
}

MONTH_PAGES_2025 = {
    "Jan 25": "0125", "Feb 25": "0225", "Mar 25": "0325", "Apr 25": "0425",
    "May 25": "0525", "Jun 25": "0625", "Jul 25": "0725", "Aug 25": "0825",
    "Sep 25": "0925", "Oct 25": "1025", "Nov 25": "1125", "Dec 25": "1225",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_month(page_code: str) -> list[dict]:
    """Fetch one month page from lottolore and parse all draws."""
    url = f"https://lottolore.com/l649{page_code}.html"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    draws = []
    i = 0
    while i < len(lines):
        # Look for date pattern like "Sat, March 21, 2026 - Lotto 6/49"
        date_match = re.match(
            r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(\w+\s+\d+,\s+\d{4})\s*[-–]\s*Lotto 6/49",
            lines[i], re.IGNORECASE
        )
        if date_match:
            date_str = date_match.group(2).strip()
            try:
                draw_date = datetime.strptime(date_str, "%B %d, %Y").date()
            except:
                i += 1
                continue
            # Scan ahead up to 20 lines for 6 numbers in a row
            found_numbers = []
            bonus = None
            for j in range(i + 1, min(i + 20, len(lines))):
                nums = re.findall(r'\b([1-9]|[1-3][0-9]|4[0-9])\b', lines[j])
                nums = [int(n) for n in nums if 1 <= int(n) <= 49]
                if len(nums) >= 6:
                    found_numbers = nums[:6]
                    # Try to find bonus on same or next line
                    bonus_match = re.search(r'[Bb]onus\s+(\d+)', lines[j])
                    if not bonus_match and j + 1 < len(lines):
                        bonus_match = re.search(r'[Bb]onus\s+(\d+)', lines[j + 1])
                    if bonus_match:
                        b = int(bonus_match.group(1))
                        if 1 <= b <= 49:
                            bonus = b
                    break
            if len(found_numbers) == 6:
                draws.append({
                    "DATE": draw_date,
                    "NUMBER DRAWN 1": found_numbers[0],
                    "NUMBER DRAWN 2": found_numbers[1],
                    "NUMBER DRAWN 3": found_numbers[2],
                    "NUMBER DRAWN 4": found_numbers[3],
                    "NUMBER DRAWN 5": found_numbers[4],
                    "NUMBER DRAWN 6": found_numbers[5],
                    "BONUS": bonus,
                })
        i += 1
    return draws

@st.cache_data(ttl=3600)
def load_all_draws(years: list[str]) -> pd.DataFrame:
    """Load draws for selected years from lottolore."""
    all_draws = []
    pages_to_fetch = []

    if "2026" in years:
        now = datetime.now()
        current_month_abbr = now.strftime("%b")
        for m, code in MONTHS.items():
            month_num = datetime.strptime(m, "%b").month
            if month_num <= now.month:
                pages_to_fetch.append(code)

    if "2025" in years:
        for label, code in MONTH_PAGES_2025.items():
            pages_to_fetch.append(code)

    progress = st.progress(0, text="Fetching draw history...")
    total = len(pages_to_fetch)

    for idx, code in enumerate(pages_to_fetch):
        draws = fetch_month(code)
        all_draws.extend(draws)
        progress.progress((idx + 1) / total, text=f"Fetching... ({idx+1}/{total} months)")
        time.sleep(0.1)

    progress.empty()

    if not all_draws:
        return pd.DataFrame()

    df = pd.DataFrame(all_draws)
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.drop_duplicates(subset=[f"NUMBER DRAWN {i}" for i in range(1, 7)] + ["DATE"])
    return df

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

NUMBER_COLS = [f"NUMBER DRAWN {i}" for i in range(1, 7)]

def frequency_table(df):
    return Counter(df[NUMBER_COLS].values.flatten().tolist())

def gap_table(df):
    last_seen = {}
    for idx, row in df.iterrows():
        for n in row[NUMBER_COLS]:
            last_seen[int(n)] = idx
    total = len(df)
    rows = []
    for n in range(1, 50):
        last = last_seen.get(n, -1)
        gap = (total - 1 - last) if last >= 0 else total
        rows.append({"Number": n, "Gap": gap})
    return pd.DataFrame(rows).sort_values("Gap", ascending=False).reset_index(drop=True)

def pair_freq(df):
    c = Counter()
    for _, row in df.iterrows():
        c.update(combinations(sorted(row[NUMBER_COLS].tolist()), 2))
    top = c.most_common(25)
    return pd.DataFrame([{"Pair": f"{a} & {b}", "Count": cnt} for (a, b), cnt in top])

def decade_breakdown(df):
    bands = {"1–9": range(1,10),"10–19": range(10,20),"20–29": range(20,30),
             "30–39": range(30,40),"40–49": range(40,50)}
    rows = []
    for label, rng in bands.items():
        s = set(rng)
        total = sum(1 for n in df[NUMBER_COLS].values.flatten() if n in s)
        rows.append({"Decade": label, "Total": total, "Avg per draw": round(total / len(df), 2)})
    return pd.DataFrame(rows)

def popularity_score(n):
    score = 1.0
    if n <= 12:   score += 0.8
    elif n <= 31: score += 0.4
    else:         score -= 0.3
    if n in [7, 3, 11, 13, 17, 21]: score += 0.3
    return round(score, 2)

def split_risk(ticket):
    avg = sum(popularity_score(n) for n in ticket) / 6
    if avg > 1.2: return "🔴 High split risk"
    if avg > 0.9: return "🟡 Medium split risk"
    return "🟢 Low split risk"

def has_consec(t):
    s = sorted(t)
    return any(s[i+1]-s[i]==1 for i in range(5))

def is_arith(t):
    s = sorted(t)
    g = [s[i+1]-s[i] for i in range(5)]
    return len(set(g)) == 1

def passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
    if not (sum_min <= sum(t) <= sum_max): return False
    if sum(1 for n in t if n > 31) < above31_min: return False
    if no_consec and has_consec(t): return False
    if no_arith and is_arith(t): return False
    odds = sum(1 for n in t if n % 2 != 0)
    if not (odd_min <= odds <= odd_max): return False
    return True

def gen_ticket(pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=8000):
    p = [int(n) for n in pool if 1 <= n <= 49]
    if len(p) < 6:
        p = list(range(1, 50))
    for _ in range(tries):
        t = sorted(random.sample(p, 6))
        if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            return t
    return sorted(random.sample(p, 6))

def decade_spread_ticket(excluded, recent, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=15000):
    skip = excluded | recent
    d1 = [n for n in range(1,10)  if n not in skip]
    d2 = [n for n in range(10,20) if n not in skip]
    d3 = [n for n in range(20,30) if n not in skip]
    hi = [n for n in range(30,50) if n not in skip]
    for _ in range(tries):
        if not d1 or not d2 or not d3 or len(hi) < 3: break
        t = sorted([random.choice(d1), random.choice(d2), random.choice(d3)] + random.sample(hi, 3))
        if len(set(t)) < 6: continue
        if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            return t
    # Relax recent exclusion if no valid ticket found
    d1 = [n for n in range(1,10)  if n not in excluded]
    d2 = [n for n in range(10,20) if n not in excluded]
    d3 = [n for n in range(20,30) if n not in excluded]
    hi = [n for n in range(30,50) if n not in excluded]
    for _ in range(tries):
        if not d1 or not d2 or not d3 or len(hi) < 3: break
        t = sorted([random.choice(d1), random.choice(d2), random.choice(d3)] + random.sample(hi, 3))
        if len(set(t)) < 6: continue
        if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            return t
    return None

def delta_ticket(delta_dist, excluded, recent, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=8000):
    skip = excluded | recent
    top_deltas = [d for d, _ in delta_dist.most_common(15)] or list(range(1,10))
    pool = [n for n in range(1,50) if n not in skip]
    for _ in range(tries):
        start = random.randint(1, 15)
        seq = [start]
        for _ in range(5):
            seq.append(seq[-1] + random.choice(top_deltas))
        seq = [n for n in seq if 1 <= n <= 49 and n not in skip]
        if len(set(seq)) == 6:
            t = sorted(seq)
            if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
                return t
    return gen_ticket(pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")

years_to_load = st.sidebar.multiselect(
    "Years to load", ["2026", "2025"], default=["2026", "2025"],
    help="More years = slower load but richer statistics."
)

if st.sidebar.button("🔄 Refresh live data", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Generator constraints")
sum_min = st.sidebar.slider("Min sum", 60, 180, 115)
sum_max = st.sidebar.slider("Max sum", 120, 250, 185)
above31_min = st.sidebar.slider("Min numbers above 31", 0, 6, 3,
    help="The only real edge: avoids birthday number splitting")
odd_min, odd_max = st.sidebar.select_slider("Odd count range", options=list(range(7)), value=(2, 4))
no_consec = st.sidebar.checkbox("No consecutive numbers", value=True)
no_arith  = st.sidebar.checkbox("No arithmetic sequences", value=True)
n_tickets = st.sidebar.slider("Tickets to generate", 1, 10, 4)

excl_str = st.sidebar.text_input("Exclude numbers (comma-separated)", "")
excluded = {int(x.strip()) for x in excl_str.split(",") if x.strip().isdigit() and 1 <= int(x.strip()) <= 49}

exclude_last_draw = st.sidebar.checkbox("Auto-exclude last draw numbers", value=True)
n_recent_to_exclude = st.sidebar.slider("Draws to exclude from pool", 1, 5, 1,
    help="Excludes numbers from the N most recent draws",
    disabled=not exclude_last_draw)

st.sidebar.markdown("---")
st.sidebar.subheader("🩷 Your lucky combo")
lucky_str = st.sidebar.text_input("Always-play numbers", "3,9,10,12,17,25")

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

if not years_to_load:
    st.warning("Select at least one year in the sidebar.")
    st.stop()

with st.spinner("Loading live draw data from the web..."):
    df = load_all_draws(years_to_load)

if df.empty:
    st.error(
        "Could not fetch draw data. Check your internet connection. "
        "Make sure `requests` and `beautifulsoup4` are installed: "
        "`pip install requests beautifulsoup4 lxml`"
    )
    st.stop()

st.success(f"✅ {len(df)} draws loaded — most recent: **{df['DATE'].iloc[-1]}**")

# ──────────────────────────────────────────────
# LAST DRAW BANNER
# ──────────────────────────────────────────────

last_row = df.iloc[-1]
last_nums = [int(last_row[c]) for c in NUMBER_COLS]
last_bonus = int(last_row["BONUS"]) if pd.notna(last_row.get("BONUS")) else None
prev_row  = df.iloc[-2]
prev_nums = [int(prev_row[c]) for c in NUMBER_COLS]

st.markdown("---")
col_l, col_r = st.columns([3, 1])
with col_l:
    st.markdown(f"**Last draw ({last_row['DATE']}):** "
                + "  ".join([f"`{n}`" for n in last_nums])
                + (f"  Bonus: `{last_bonus}`" if last_bonus else ""))
    st.markdown(f"**Previous ({prev_row['DATE']}):** "
                + "  ".join([f"`{n}`" for n in prev_nums]))
with col_r:
    next_draw_day = "Wednesday" if datetime.now().weekday() < 2 else "Saturday"
    st.info(f"Next draw: **{next_draw_day}**")

st.markdown("---")

# Build recent exclusion set
recent_set = set()
if exclude_last_draw:
    for _, row in df.tail(n_recent_to_exclude).iterrows():
        recent_set.update(int(row[c]) for c in NUMBER_COLS)

# Precompute analytics
freq       = frequency_table(df)
gaps_df    = gap_table(df)
hot6       = [n for n, _ in freq.most_common(6)]
cold6      = [n for n, _ in freq.most_common()[:-7:-1]]
overdue15  = gaps_df.head(15)["Number"].tolist()
full_pool  = [n for n in range(1, 50) if n not in excluded]
fresh_pool = [n for n in range(1, 50) if n not in excluded and n not in recent_set]
hot_pool   = [n for n, _ in freq.most_common(20) if n not in excluded and n not in recent_set]
cold_pool  = [n for n, _ in freq.most_common()[:-21:-1] if n not in excluded and n not in recent_set]
over_pool  = [n for n in overdue15 if n not in excluded and n not in recent_set]

delta_dist = Counter()
for _, row in df.iterrows():
    s = sorted(row[NUMBER_COLS].tolist())
    delta_dist.update(s[i+1]-s[i] for i in range(5))

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

tab_gen, tab_lucky, tab_freq, tab_gaps, tab_dec, tab_pairs, tab_check = st.tabs([
    "🎟️ Generate tickets", "🩷 Lucky combo", "📊 Frequency",
    "⏳ Gap analysis", "🔢 Decades", "🤝 Pairs", "🔍 Check a combo"
])

# ── GENERATORS ──────────────────────────────
with tab_gen:
    st.subheader("Ticket generator")

    if exclude_last_draw and recent_set:
        st.info(f"Auto-excluding from pool: `{sorted(recent_set)}` (last {n_recent_to_exclude} draw{'s' if n_recent_to_exclude>1 else ''})")

    gen_mode = st.selectbox("Strategy", [
        "Decade spread — recommended",
        "Filtered random",
        "Hot numbers biased",
        "Cold numbers biased",
        "Overdue numbers biased",
        "Delta system",
        "Gut pick (unpopular territory)",
    ])

    show_lucky = st.checkbox("Include my lucky combo as Draw 1", value=True)

    if st.button("🎲 Generate", type="primary"):

        # Parse lucky combo
        lucky = []
        if show_lucky and lucky_str.strip():
            try:
                lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
                if len(lucky) != 6: lucky = []
            except: lucky = []

        tickets = []
        for _ in range(n_tickets):
            if gen_mode == "Decade spread — recommended":
                t = decade_spread_ticket(excluded, recent_set, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Filtered random":
                t = gen_ticket(fresh_pool or full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Hot numbers biased":
                t = gen_ticket(hot_pool or full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Cold numbers biased":
                t = gen_ticket(cold_pool or full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Overdue numbers biased":
                t = gen_ticket((over_pool + full_pool) or full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Delta system":
                t = delta_ticket(delta_dist, excluded, recent_set, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            elif gen_mode == "Gut pick (unpopular territory)":
                gut_pool = [n for n in range(32, 50) if n not in excluded and n not in recent_set]
                gut_low  = [n for n in [2,4,5,6,7,8,13,14,15,19,23,26,27,29] if n not in excluded and n not in recent_set]
                combined = gut_low + gut_pool
                t = gen_ticket(combined or fresh_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            else:
                t = gen_ticket(full_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            if t:
                tickets.append(t)

        draw_num = 1

        # Show lucky combo first
        if lucky:
            s = sum(lucky)
            above = sum(1 for n in lucky if n > 31)
            odds  = sum(1 for n in lucky if n % 2 != 0)
            in_recent = [n for n in lucky if n in recent_set]
            risk  = split_risk(lucky)
            st.markdown(f"---")
            st.markdown(f"**🩷 Draw {draw_num} — Lucky combo:** `{lucky}`")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Sum", s, delta="OK" if 115<=s<=185 else "⚠️ Low")
            c2.metric("Above 31", f"{above}/6")
            c3.metric("Odd/Even", f"{odds}/{6-odds}")
            c4.metric("Split risk", risk.split()[0])
            if in_recent:
                st.warning(f"⚠️ {in_recent} appeared in the last {n_recent_to_exclude} draw(s)")
            draw_num += 1

        # Show generated tickets
        for t in tickets:
            s = sum(t)
            above = sum(1 for n in t if n > 31)
            odds  = sum(1 for n in t if n % 2 != 0)
            risk  = split_risk(t)
            overlap = [n for n in t if n in recent_set]
            st.markdown(f"---")
            st.markdown(f"**Draw {draw_num}:** `{t}`")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Sum", s)
            c2.metric("Above 31", f"{above}/6")
            c3.metric("Odd/Even", f"{odds}/{6-odds}")
            c4.metric("Split risk", risk.split()[0])
            if overlap:
                st.warning(f"⚠️ {overlap} appeared in last draw(s) — consider regenerating")
            draw_num += 1

# ── LUCKY COMBO ──────────────────────────────
with tab_lucky:
    st.subheader("🩷 Lucky combo deep audit")
    if lucky_str.strip():
        try:
            lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
            if len(lucky) == 6:
                s     = sum(lucky)
                above = sum(1 for n in lucky if n > 31)
                odds  = sum(1 for n in lucky if n % 2 != 0)
                risk  = split_risk(lucky)
                in_recent = [n for n in lucky if n in recent_set]

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Sum", s, delta="OK" if 115<=s<=185 else "⚠️ Out of range")
                c2.metric("Above 31", f"{above}/6", delta="OK" if above>=3 else "⚠️ Too low")
                c3.metric("Odd/Even", f"{odds}/{6-odds}")
                c4.metric("Consecutive?", "Yes ⚠️" if has_consec(lucky) else "No ✓")

                st.write(f"**Split risk:** {risk}")

                if in_recent:
                    st.warning(f"⚠️ Numbers {in_recent} appeared in the last {n_recent_to_exclude} draw(s).")

                ctx = pd.DataFrame({
                    "Number": lucky,
                    "All-time freq": [freq.get(n, 0) for n in lucky],
                    "Draws since last seen": [int(gaps_df.set_index("Number").loc[n, "Gap"]) for n in lucky],
                    "Popularity score": [popularity_score(n) for n in lucky],
                    "Above 31": ["Yes" if n > 31 else "No" for n in lucky],
                    "In recent draw?": ["🔴 Yes" if n in recent_set else "No" for n in lucky],
                })
                st.dataframe(ctx, use_container_width=True)

                if s < 115:
                    st.error(f"Sum {s} is well below the 115–185 range. All numbers cluster too low.")
                if above == 0:
                    st.error("Zero numbers above 31 — entirely birthday territory. Very high split risk if you win.")
        except Exception as e:
            st.warning(f"Could not parse: {e}")

# ── FREQUENCY ────────────────────────────────
with tab_freq:
    st.subheader("Number frequency")
    freq_df = pd.DataFrame({"Number": list(range(1,50)), "Frequency": [freq.get(n,0) for n in range(1,50)]})
    fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                 color_continuous_scale="Blues", title="All-time frequency per number")
    st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    c1.metric("🔥 Top 6 hot", str(hot6))
    c2.metric("❄️ Bottom 6 cold", str(cold6))

# ── GAP ANALYSIS ─────────────────────────────
with tab_gaps:
    st.subheader("Gap analysis — draws since last appearance")
    fig2 = px.bar(gaps_df, x="Number", y="Gap", color="Gap",
                  color_continuous_scale="Oranges",
                  title="Draws since each number was last seen")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(gaps_df.head(15), use_container_width=True)
    st.caption("⚠️ Overdue does NOT mean more likely. The machine has no memory. For exploration only.")

# ── DECADE BREAKDOWN ─────────────────────────
with tab_dec:
    st.subheader("Decade breakdown")
    dec_df = decade_breakdown(df)
    fig3 = px.bar(dec_df, x="Decade", y="Avg per draw", color="Avg per draw",
                  color_continuous_scale="Teal", text="Avg per draw",
                  title="Average numbers drawn per decade range")
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(dec_df, use_container_width=True)

# ── PAIRS ────────────────────────────────────
with tab_pairs:
    st.subheader("Most common pairs")
    p_df = pair_freq(df)
    fig4 = px.bar(p_df, x="Count", y="Pair", orientation="h",
                  color="Count", color_continuous_scale="Viridis",
                  title="Top 25 most co-occurring pairs")
    fig4.update_layout(yaxis={"categoryorder":"total ascending"}, height=600)
    st.plotly_chart(fig4, use_container_width=True)

# ── CHECK COMBO ──────────────────────────────
with tab_check:
    st.subheader("Has this combination ever been drawn?")
    check_str = st.text_input("6 numbers comma-separated:", "")
    if check_str.strip():
        try:
            check = tuple(sorted(int(x.strip()) for x in check_str.split(",") if x.strip().isdigit()))
            if len(check) != 6:
                st.warning("Enter exactly 6 numbers.")
            else:
                past = [tuple(sorted(row[NUMBER_COLS].tolist())) for _, row in df.iterrows()]
                count = past.count(check)
                if count > 0:
                    st.success(f"✅ Appeared {count} time(s) in {len(df)} draws.")
                else:
                    st.info(f"❌ Never appeared in {len(df)} draws analyzed.")
        except:
            st.warning("Invalid input.")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**The honest truth:** Lotto 6/49 is a certified random draw. "
    "Your odds are 1 in 13,983,816 per ticket, every draw, forever — regardless of any strategy. "
    "The only real edge is split avoidance: pick unpopular numbers above 31 so if you win, you keep more of it. "
    "Data sourced from lottolore.com · Refreshes every hour."
)
