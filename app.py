import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from collections import Counter
from itertools import combinations
import random
import re
from datetime import datetime, date

st.set_page_config(page_title="Lotto 6/49 Live Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Lotto 6/49 — Live Analyzer")
st.caption(
    "Pulls real draw results live from the web. No CSV. No ML. No snake oil. "
    "Honest statistics + split-avoidance logic. Every combo = 1 in 13,983,816 odds."
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-CA,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

NUMBER_COLS = [f"NUMBER DRAWN {i}" for i in range(1, 7)]

# Regex: longest match first to avoid '1' matching before '19'
NUM_RE = re.compile(r'\b(4[0-9]|[1-3][0-9]|[1-9])\b')


# ──────────────────────────────────────────────
# SCRAPING
# ──────────────────────────────────────────────

def _extract_balls_from_text(text: str, exclude_ints: set[int] | None = None) -> list[int]:
    """
    Extract 6 distinct lotto balls (1–49) from a text block.

    exclude_ints: a set of integers to strip before picking balls.
    Used to remove date components (year, month number, day) that
    commonly bleed into the candidate pool and corrupt results.

    Returns a list of exactly 6 ints, or [] if extraction fails.
    """
    if exclude_ints is None:
        exclude_ints = set()

    raw = NUM_RE.findall(text)
    seen: set[int] = set()
    candidates: list[int] = []
    for s in raw:
        n = int(s)
        if n not in seen and n not in exclude_ints:
            seen.add(n)
            candidates.append(n)

    if len(candidates) >= 6:
        return candidates[:6]
    return []


def _date_exclusions(d: date) -> set[int]:
    """
    Return the set of integers that appear in a date's numeric representation
    and must be excluded from ball extraction to avoid contamination.

    e.g. 2026-04-19  →  {2026, 26, 4, 19}  filtered to 1–49  →  {4, 19, 26}
    """
    parts = {d.year, d.month, d.day, d.year % 100}
    return {p for p in parts if 1 <= p <= 49}


def _session() -> requests.Session:
    """Return a requests Session with browser-like headers."""
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_draw_dates_for_year(sess: requests.Session, year: int) -> list[date]:
    """
    Fetch the draw-date index page for a given year and return all individual
    draw dates found as href slugs.
    """
    url = f"https://www.lottoresult.ca/games/lotto-649/results/{year}"
    try:
        r = sess.get(url, timeout=20)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    pattern = re.compile(rf"/games/lotto-649/results/{year}-(\d{{2}})-(\d{{2}})$")
    dates: list[date] = []
    seen: set[str] = set()
    for link in soup.find_all("a", href=pattern):
        href = link["href"]
        if href in seen:
            continue
        seen.add(href)
        m = re.search(r"(\d{4}-\d{2}-\d{2})$", href)
        if m:
            try:
                dates.append(datetime.strptime(m.group(1), "%Y-%m-%d").date())
            except ValueError:
                pass
    return sorted(set(dates))


def fetch_draw_result(sess: requests.Session, draw_date: date) -> dict | None:
    """
    Fetch a single draw's result page and extract the 6 winning balls + bonus.

    Strategy:
    1. Build the per-draw URL.
    2. Parse the page and look for the smallest DOM container that holds
       exactly 6 or 7 unique numbers in 1–49 after excluding date digits.
    3. Fall back to full-page extraction if no clean container found.
    4. Validate the result strictly (6 distinct numbers, all 1–49).
    """
    url = f"https://www.lottoresult.ca/games/lotto-649/results/{draw_date}"
    try:
        r = sess.get(url, timeout=15)
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "lxml")
    exclusions = _date_exclusions(draw_date)

    # ── Strategy 1: find a tight container (div/ul/section/table) ──────────
    # We want the smallest element whose text yields exactly 6–7 unique
    # 1–49 numbers (6 main balls + optional bonus).
    candidates_by_size: list[tuple[int, list[int], int | None]] = []

    for tag in soup.find_all(["div", "ul", "ol", "section", "table", "span"]):
        text = tag.get_text(separator=" ")
        balls = _extract_balls_from_text(text, exclusions)
        if len(balls) >= 6:
            # Score by: fewest extra numbers (tighter container = better)
            all_nums = NUM_RE.findall(text)
            extra = len(all_nums) - 6
            bonus = balls[6] if len(balls) >= 7 else None
            candidates_by_size.append((extra, balls[:6], bonus, len(text)))

    if candidates_by_size:
        # Sort by (fewest extra numbers, shortest text) — tightest container wins
        candidates_by_size.sort(key=lambda x: (x[0], x[3]))
        _, balls, bonus, _ = candidates_by_size[0]
        if len(set(balls)) == 6:
            return _build_row(draw_date, balls, bonus)

    # ── Strategy 2: full page text with exclusions ──────────────────────────
    text = soup.get_text(separator=" ")
    balls = _extract_balls_from_text(text, exclusions)
    if len(set(balls)) == 6:
        return _build_row(draw_date, balls, None)

    return None


def _build_row(draw_date: date, balls: list[int], bonus: int | None) -> dict:
    return {
        "DATE": draw_date,
        "NUMBER DRAWN 1": balls[0],
        "NUMBER DRAWN 2": balls[1],
        "NUMBER DRAWN 3": balls[2],
        "NUMBER DRAWN 4": balls[3],
        "NUMBER DRAWN 5": balls[4],
        "NUMBER DRAWN 6": balls[5],
        "BONUS": bonus,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_draws(n_years: int) -> pd.DataFrame:
    """
    Two-phase load:
    Phase 1 – collect all draw dates from year-index pages.
    Phase 2 – fetch each individual draw page for clean, isolated extraction.

    This avoids the DOM-walking approach that contaminated results with
    navigation numbers and date digits.
    """
    sess = _session()
    current_year = datetime.now().year
    years = list(range(current_year, current_year - n_years, -1))

    # ── Phase 1: collect dates ───────────────────────────────────────────
    all_dates: list[date] = []
    date_progress = st.progress(0, text="Fetching draw date index...")
    for idx, year in enumerate(years):
        dates = fetch_draw_dates_for_year(sess, year)
        all_dates.extend(dates)
        date_progress.progress((idx + 1) / len(years), text=f"Indexed {year}: {len(dates)} draws found")
    date_progress.empty()

    all_dates = sorted(set(all_dates))
    if not all_dates:
        return pd.DataFrame()

    # ── Phase 2: fetch individual results ───────────────────────────────
    all_draws: list[dict] = []
    result_progress = st.progress(0, text="Fetching individual draw results...")
    failed = 0
    for idx, draw_date in enumerate(all_dates):
        row = fetch_draw_result(sess, draw_date)
        if row:
            all_draws.append(row)
        else:
            failed += 1
        pct = (idx + 1) / len(all_dates)
        result_progress.progress(
            pct,
            text=f"Loaded {idx + 1}/{len(all_dates)} draws  ({failed} failed)"
        )
    result_progress.empty()

    if failed > 0:
        st.warning(
            f"⚠️ {failed} draw(s) could not be parsed and were skipped. "
            "Try **Refresh live data** to retry."
        )

    if not all_draws:
        return pd.DataFrame()

    df = pd.DataFrame(all_draws)
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.drop_duplicates(subset=NUMBER_COLS + ["DATE"]).reset_index(drop=True)

    # ── Validate: flag rows where numbers are out of range or duplicated ─
    def _is_valid_row(row):
        nums = [row[c] for c in NUMBER_COLS]
        return (
            len(set(nums)) == 6
            and all(isinstance(n, (int, float)) and 1 <= int(n) <= 49 for n in nums)
        )

    valid_mask = df.apply(_is_valid_row, axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        st.warning(f"⚠️ {n_invalid} draw(s) had invalid numbers and were removed.")
        df = df[valid_mask].reset_index(drop=True)

    return df


# ──────────────────────────────────────────────
# ANALYTICS HELPERS
# ──────────────────────────────────────────────

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
    bands = {
        "1–9":   range(1, 10),
        "10–19": range(10, 20),
        "20–29": range(20, 30),
        "30–39": range(30, 40),
        "40–49": range(40, 50),
    }
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
    if n in [3, 7, 11, 13, 17, 21]: score += 0.3
    return round(score, 2)

def split_risk(ticket):
    avg = sum(popularity_score(n) for n in ticket) / 6
    if avg > 1.2: return "🔴 High split risk"
    if avg > 0.9: return "🟡 Medium split risk"
    return "🟢 Low split risk"

def has_consec(t):
    s = sorted(t)
    return any(s[i + 1] - s[i] == 1 for i in range(5))

def is_arith(t):
    s = sorted(t)
    g = [s[i + 1] - s[i] for i in range(5)]
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
    if len(p) < 6: p = list(range(1, 50))
    for _ in range(tries):
        t = sorted(random.sample(p, 6))
        if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            return t
    return sorted(random.sample(p, 6))

def decade_spread_ticket(excluded, recent, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=15000):
    for attempt in range(2):
        skip = (excluded | recent) if attempt == 0 else excluded
        d1 = [n for n in range(1, 10)  if n not in skip]
        d2 = [n for n in range(10, 20) if n not in skip]
        d3 = [n for n in range(20, 30) if n not in skip]
        hi = [n for n in range(30, 50) if n not in skip]
        for _ in range(tries):
            if not d1 or not d2 or not d3 or len(hi) < 3: break
            t = sorted([random.choice(d1), random.choice(d2), random.choice(d3)] + random.sample(hi, 3))
            if len(set(t)) < 6: continue
            if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
                return t
    return None

def delta_ticket(delta_dist, excluded, recent, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=8000):
    skip = excluded | recent
    top_deltas = [d for d, _ in delta_dist.most_common(15)] or list(range(1, 10))
    pool = [n for n in range(1, 50) if n not in skip]
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
    return gen_ticket(pool or list(range(1, 50)), sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")

n_years = st.sidebar.slider("Years of history to load", 1, 5, 2,
    help="1 year ≈ 104 draws. More years = richer stats, slower first load.")

if st.sidebar.button("🔄 Refresh live data", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Generator rules")
sum_min     = st.sidebar.slider("Min sum", 60, 180, 115)
sum_max     = st.sidebar.slider("Max sum", 120, 250, 185)
above31_min = st.sidebar.slider("Min numbers above 31", 0, 6, 3,
    help="The only real edge — avoids jackpot splitting")
odd_min, odd_max = st.sidebar.select_slider(
    "Odd count range", options=list(range(7)), value=(2, 4))
no_consec = st.sidebar.checkbox("No consecutive numbers", value=True)
no_arith  = st.sidebar.checkbox("No arithmetic sequences", value=True)
n_tickets = st.sidebar.slider("Tickets to generate", 1, 10, 4)

excl_str = st.sidebar.text_input("Exclude numbers (comma-separated)", "")
excluded = {int(x.strip()) for x in excl_str.split(",")
            if x.strip().isdigit() and 1 <= int(x.strip()) <= 49}

exclude_recent   = st.sidebar.checkbox("Auto-exclude last draw numbers", value=True)
n_recent_exclude = st.sidebar.slider("Recent draws to exclude", 1, 5, 1, disabled=not exclude_recent)

st.sidebar.markdown("---")
st.sidebar.subheader("🩷 Lucky combo")
lucky_str = st.sidebar.text_input("Your always-play numbers", "3,9,10,12,17,25")


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

with st.spinner("Loading live draw results from lottoresult.ca..."):
    df = load_draws(n_years)

if df.empty:
    st.error(
        "❌ Could not load draw data from lottoresult.ca. "
        "The site may be temporarily unavailable or blocking automated requests. "
        "Try clicking **Refresh live data** in the sidebar, or wait a few minutes and try again."
    )
    st.stop()

st.success(f"✅ {len(df)} draws loaded — most recent: **{df['DATE'].iloc[-1]}**")

# ── Last draw banner ──────────────────────────
last_row   = df.iloc[-1]
last_nums  = [int(last_row[c]) for c in NUMBER_COLS]
last_bonus = int(last_row["BONUS"]) if pd.notna(last_row.get("BONUS")) else None
prev_row   = df.iloc[-2]
prev_nums  = [int(prev_row[c]) for c in NUMBER_COLS]

st.markdown("---")
col_a, col_b = st.columns([3, 1])
with col_a:
    balls_str = "  ".join([f"`{n}`" for n in last_nums])
    bonus_str = f"  +Bonus `{last_bonus}`" if last_bonus else ""
    st.markdown(f"**Last draw ({last_row['DATE']}):** {balls_str}{bonus_str}")
    prev_str = "  ".join([f"`{n}`" for n in prev_nums])
    st.markdown(f"**Previous ({prev_row['DATE']}):** {prev_str}")
with col_b:
    weekday = datetime.now().weekday()
    next_draw = "Wednesday" if weekday in [3, 4, 5, 6, 0] else "Saturday"
    st.info(f"Next draw: **{next_draw}**")
st.markdown("---")

# ── Build pools ───────────────────────────────
recent_set = set()
if exclude_recent:
    for _, row in df.tail(n_recent_exclude).iterrows():
        recent_set.update(int(row[c]) for c in NUMBER_COLS)

freq      = frequency_table(df)
gaps_df   = gap_table(df)
hot6      = [n for n, _ in freq.most_common(6)]
cold6     = [n for n, _ in freq.most_common()[:-7:-1]]
over15    = gaps_df.head(15)["Number"].tolist()
full_pool  = [n for n in range(1, 50) if n not in excluded]
fresh_pool = [n for n in range(1, 50) if n not in excluded and n not in recent_set]
hot_pool   = [n for n, _ in freq.most_common(20) if n not in excluded and n not in recent_set]
cold_pool  = [n for n, _ in freq.most_common()[:-21:-1] if n not in excluded and n not in recent_set]
over_pool  = [n for n in over15 if n not in excluded and n not in recent_set]

delta_dist = Counter()
for _, row in df.iterrows():
    s = sorted(row[NUMBER_COLS].tolist())
    delta_dist.update(s[i + 1] - s[i] for i in range(5))


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

tab_gen, tab_lucky, tab_freq_t, tab_gaps_t, tab_dec, tab_pairs, tab_check, tab_hist = st.tabs([
    "🎟️ Generate", "🩷 Lucky combo", "📊 Frequency",
    "⏳ Gaps", "🔢 Decades", "🤝 Pairs", "🔍 Check combo", "📋 History"
])

# ── GENERATE ──────────────────────────────────
with tab_gen:
    st.subheader("Ticket generator")
    if exclude_recent and recent_set:
        st.info(f"Auto-excluding (last {n_recent_exclude} draw{'s' if n_recent_exclude > 1 else ''}): `{sorted(recent_set)}`")

    gen_mode   = st.selectbox("Strategy", [
        "Decade spread — recommended", "Filtered random",
        "Hot numbers biased", "Cold numbers biased",
        "Overdue numbers biased", "Delta system",
        "Gut pick (unpopular territory)",
    ])
    show_lucky = st.checkbox("Include my lucky combo as Draw 1", value=True)

    if st.button("🎲 Generate tickets", type="primary"):
        lucky = []
        if show_lucky and lucky_str.strip():
            try:
                lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
                if len(lucky) != 6: lucky = []
            except Exception: lucky = []

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
            else:
                gut = [n for n in [2, 4, 5, 6, 7, 8, 13, 14, 15, 19, 23, 26, 27, 29, 32, 33, 35, 37, 39, 42, 43, 44, 47, 48]
                       if n not in excluded and n not in recent_set]
                t = gen_ticket(gut or fresh_pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max)
            if t: tickets.append(t)

        draw_num = 1
        if lucky:
            s = sum(lucky); above = sum(1 for n in lucky if n > 31); odds = sum(1 for n in lucky if n % 2 != 0)
            in_rec = [n for n in lucky if n in recent_set]
            st.markdown("---")
            st.markdown(f"**🩷 Draw {draw_num} — Lucky combo:** `{lucky}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sum", s, delta="OK ✓" if 115 <= s <= 185 else "⚠️ Low")
            c2.metric("Above 31", f"{above}/6", delta="OK ✓" if above >= 3 else "⚠️ Low")
            c3.metric("Odd/Even", f"{odds}/{6 - odds}")
            c4.metric("Split risk", split_risk(lucky).split()[0])
            if in_rec: st.warning(f"⚠️ {in_rec} appeared in the last draw(s)")
            draw_num += 1

        for t in tickets:
            s = sum(t); above = sum(1 for n in t if n > 31); odds = sum(1 for n in t if n % 2 != 0)
            in_rec = [n for n in t if n in recent_set]
            st.markdown("---")
            st.markdown(f"**Draw {draw_num}:** `{t}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sum", s); c2.metric("Above 31", f"{above}/6")
            c3.metric("Odd/Even", f"{odds}/{6 - odds}"); c4.metric("Split risk", split_risk(t).split()[0])
            if in_rec: st.warning(f"⚠️ {in_rec} appeared in last draw(s) — consider regenerating")
            draw_num += 1

# ── LUCKY COMBO ───────────────────────────────
with tab_lucky:
    st.subheader("🩷 Lucky combo deep audit")
    try:
        lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
        if len(lucky) == 6:
            s = sum(lucky); above = sum(1 for n in lucky if n > 31); odds = sum(1 for n in lucky if n % 2 != 0)
            in_rec = [n for n in lucky if n in recent_set]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sum", s, delta="OK ✓" if 115 <= s <= 185 else "⚠️ Out of range")
            c2.metric("Above 31", f"{above}/6", delta="OK ✓" if above >= 3 else "⚠️ Too low")
            c3.metric("Odd/Even", f"{odds}/{6 - odds}")
            c4.metric("Consecutive?", "Yes ⚠️" if has_consec(lucky) else "No ✓")
            st.write(f"**Split risk:** {split_risk(lucky)}")
            if in_rec: st.warning(f"⚠️ {in_rec} appeared in the last {n_recent_exclude} draw(s).")
            ctx = pd.DataFrame({
                "Number": lucky,
                "All-time frequency": [freq.get(n, 0) for n in lucky],
                "Draws since last seen": [int(gaps_df.set_index("Number").loc[n, "Gap"]) for n in lucky],
                "Popularity score": [popularity_score(n) for n in lucky],
                "Above 31": ["Yes" if n > 31 else "No" for n in lucky],
                "In recent draw?": ["🔴 Yes" if n in recent_set else "No" for n in lucky],
            })
            st.dataframe(ctx, use_container_width=True)
            if s < 115: st.error(f"Sum {s} is well below the 115–185 historical range.")
            if above == 0: st.error("Zero numbers above 31 — entirely birthday territory. Very high split risk.")
            if has_consec(lucky): st.warning("Contains consecutive numbers.")
    except Exception as e:
        st.warning(f"Could not parse lucky numbers: {e}")

# ── FREQUENCY ─────────────────────────────────
with tab_freq_t:
    st.subheader("Number frequency")
    freq_df = pd.DataFrame({"Number": list(range(1, 50)), "Frequency": [freq.get(n, 0) for n in range(1, 50)]})
    fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                 color_continuous_scale="Blues", title="All-time frequency per number")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("🔥 Top 6 hot", str(hot6)); c2.metric("❄️ Bottom 6 cold", str(cold6))

# ── GAPS ──────────────────────────────────────
with tab_gaps_t:
    st.subheader("Gap analysis")
    fig2 = px.bar(gaps_df, x="Number", y="Gap", color="Gap",
                  color_continuous_scale="Oranges", title="Draws since last appearance")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(gaps_df.head(15), use_container_width=True)
    st.caption("⚠️ Overdue does NOT mean more likely. The machine has no memory.")

# ── DECADES ───────────────────────────────────
with tab_dec:
    st.subheader("Decade breakdown")
    dec_df = decade_breakdown(df)
    fig3 = px.bar(dec_df, x="Decade", y="Avg per draw", color="Avg per draw",
                  color_continuous_scale="Teal", text="Avg per draw", title="Decade breakdown")
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(dec_df, use_container_width=True)

# ── PAIRS ─────────────────────────────────────
with tab_pairs:
    st.subheader("Most common pairs")
    p_df = pair_freq(df)
    fig4 = px.bar(p_df, x="Count", y="Pair", orientation="h",
                  color="Count", color_continuous_scale="Viridis", title="Top 25 most co-occurring pairs")
    fig4.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig4, use_container_width=True)

# ── CHECK COMBO ───────────────────────────────
with tab_check:
    st.subheader("Has this combination ever been drawn?")
    check_str = st.text_input("Enter 6 numbers (comma-separated):", "")
    if check_str.strip():
        try:
            check = tuple(sorted(int(x.strip()) for x in check_str.split(",") if x.strip().isdigit()))
            if len(check) != 6:
                st.warning("Enter exactly 6 numbers.")
            else:
                past = [tuple(sorted(int(row[c]) for c in NUMBER_COLS)) for _, row in df.iterrows()]
                count = past.count(check)
                if count > 0:
                    st.success(f"✅ Appeared {count} time(s) in {len(df)} draws.")
                else:
                    st.info(f"❌ Never appeared in {len(df)} draws analyzed.")
        except Exception:
            st.warning("Invalid input.")

# ── HISTORY ───────────────────────────────────
with tab_hist:
    st.subheader("Full draw history")
    display = df.copy()
    display["DATE"] = display["DATE"].astype(str)
    st.dataframe(
        display[["DATE"] + NUMBER_COLS + ["BONUS"]].sort_values("DATE", ascending=False),
        use_container_width=True,
        height=500,
    )

st.markdown("---")
st.caption(
    "**Honest disclaimer:** Lotto 6/49 is a certified random draw. "
    "Your odds are 1 in 13,983,816 per ticket, every draw, forever. "
    "The only real edge is split avoidance — unpopular numbers above 31. "
    "Data: lottoresult.ca · Cache refreshes every hour."
)
