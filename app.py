import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from collections import Counter
from itertools import combinations
import random
import re
import numpy as np
from datetime import datetime, date

st.set_page_config(page_title="Lotto 6/49 Live Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Lotto 6/49 — Live Analyzer")
st.caption(
    "Pulls real draw results live from ca.lottonumbers.com. No CSV. No ML. No snake oil. "
    "Honest statistics + split-avoidance logic. Every combo = 1 in 13,983,816 odds."
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

BASE_URL = "https://ca.lottonumbers.com"
PAST_URL = f"{BASE_URL}/lotto-649/past-numbers"
YEAR_URL = f"{BASE_URL}/lotto-649/numbers"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-CA,en-US;q=0.9,en;q=0.8",
}

NUMBER_COLS = [f"NUMBER DRAWN {i}" for i in range(1, 7)]

# Scoring weights — must sum to 1.0
W_SPLIT   = 0.40   # split-avoidance  (numbers >31, unpopular combos)
W_GAP     = 0.25   # historical gap balance
W_DECADE  = 0.15   # decade distribution
W_ODD     = 0.10   # odd/even balance
W_PAIR    = 0.10   # pair/triple avoidance

CANDIDATES = 100_000   # tickets evaluated per run
TOP_N      = 10        # best tickets returned


# ──────────────────────────────────────────────
# SCRAPING
# ──────────────────────────────────────────────

def _fetch(url: str) -> BeautifulSoup | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None


def _parse_date_text(text: str) -> date | None:
    text = re.sub(r'\s+', ' ', text.strip())
    for fmt in ("%A %B %d %Y", "%B %d %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    m = re.search(
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})\s+(\d{4})',
        text, re.IGNORECASE
    )
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
        except ValueError:
            pass
    return None


def _parse_draws_from_soup(soup: BeautifulSoup) -> list[dict]:
    draws: list[dict] = []
    seen_dates: set[date] = set()

    for ul in soup.find_all("ul"):
        lis = ul.find_all("li", recursive=False)
        if len(lis) < 6:
            lis = ul.find_all("li")
        if len(lis) < 6:
            continue

        nums = []
        for li in lis:
            t = li.get_text(strip=True)
            t = re.match(r'^\d+', t)
            if not t:
                break
            n = int(t.group())
            if not (1 <= n <= 49):
                break
            nums.append(n)

        if len(nums) < 6:
            continue

        balls = nums[:6]
        bonus = nums[6] if len(nums) >= 7 else None

        if len(set(balls)) != 6:
            continue

        draw_date = None
        node = ul.parent
        for _ in range(8):
            if node is None:
                break
            text = node.get_text(separator=" ", strip=True)
            d = _parse_date_text(text)
            if d:
                draw_date = d
                break
            for sib in node.find_all(string=True):
                d = _parse_date_text(sib.strip())
                if d:
                    draw_date = d
                    break
            if draw_date:
                break
            node = node.parent

        if draw_date is None or draw_date in seen_dates:
            continue

        seen_dates.add(draw_date)
        draws.append({
            "DATE": draw_date,
            "NUMBER DRAWN 1": balls[0],
            "NUMBER DRAWN 2": balls[1],
            "NUMBER DRAWN 3": balls[2],
            "NUMBER DRAWN 4": balls[3],
            "NUMBER DRAWN 5": balls[4],
            "NUMBER DRAWN 6": balls[5],
            "BONUS": bonus,
        })

    return draws


@st.cache_data(ttl=3600, show_spinner=False)
def load_draws(n_years: int) -> pd.DataFrame:
    all_draws: list[dict] = []
    current_year = datetime.now().year
    years = list(range(current_year, current_year - n_years, -1))

    progress = st.progress(0, text="Fetching recent results...")
    soup = _fetch(PAST_URL)
    if soup:
        all_draws.extend(_parse_draws_from_soup(soup))

    for idx, year in enumerate(years):
        progress.progress(
            (idx + 1) / (len(years) + 1),
            text=f"Fetching {year} archive..."
        )
        soup = _fetch(f"{YEAR_URL}/{year}")
        if soup:
            all_draws.extend(_parse_draws_from_soup(soup))

    progress.empty()

    if not all_draws:
        return pd.DataFrame()

    df = pd.DataFrame(all_draws)
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.drop_duplicates(subset=["DATE"]).reset_index(drop=True)
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


# ──────────────────────────────────────────────
# SCORING ENGINE
# ──────────────────────────────────────────────

def build_scoring_tables(df, freq, gaps_df):
    """
    Pre-compute per-number and per-pair lookup tables used by score_ticket().
    Call once after data loads; pass results into score_ticket to avoid
    recomputation inside the hot loop.

    Returns a dict of lookup tables.
    """
    n_draws = len(df)

    # ── 1. Split-avoidance: popularity score per number (0–1, lower = better)
    # We invert so that unpopular (high >31) numbers score high.
    pop_raw = {n: popularity_score(n) for n in range(1, 50)}
    pop_max = max(pop_raw.values())
    pop_min = min(pop_raw.values())
    # Normalise and invert: score=1 means maximally unpopular (good for split avoid)
    split_score = {
        n: 1.0 - (pop_raw[n] - pop_min) / (pop_max - pop_min)
        for n in range(1, 50)
    }

    # ── 2. Gap score per number: ideal gap ≈ 49/6 ≈ 8 draws
    # Score peaks at the ideal gap and falls off symmetrically.
    ideal_gap = n_draws / 6  # expected average gap
    gap_by_num = gaps_df.set_index("Number")["Gap"].to_dict()
    # Distance from ideal — normalise so 0 distance → score 1
    max_possible_gap = n_draws
    gap_score = {
        n: max(0.0, 1.0 - abs(gap_by_num.get(n, ideal_gap) - ideal_gap) / max_possible_gap)
        for n in range(1, 50)
    }

    # ── 3. Decade distribution score: computed per-ticket, not per-number
    # Target: ≥1 number per decade.  Pre-compute decade membership.
    decade_map = {}
    for n in range(1, 50):
        if n <= 9:    decade_map[n] = 0
        elif n <= 19: decade_map[n] = 1
        elif n <= 29: decade_map[n] = 2
        elif n <= 39: decade_map[n] = 3
        else:         decade_map[n] = 4

    # ── 4. Pair/triple frequency — normalise count to 0–1 (higher count = more popular = worse)
    pair_counts: Counter = Counter()
    for _, row in df.iterrows():
        pair_counts.update(combinations(sorted(row[NUMBER_COLS].tolist()), 2))
    max_pair = max(pair_counts.values()) if pair_counts else 1
    # pair_penalty[pair] → 0 (never seen together) … 1 (most common pair)
    pair_penalty = {pair: cnt / max_pair for pair, cnt in pair_counts.items()}

    return {
        "split_score": split_score,
        "gap_score":   gap_score,
        "decade_map":  decade_map,
        "pair_penalty": pair_penalty,
        "n_draws": n_draws,
    }


def score_ticket(ticket: list[int], tables: dict) -> tuple[float, dict]:
    """
    Score a 6-number ticket on 0–100 scale using the weighted rubric:
        40% split-avoidance
        25% gap balance
        15% decade distribution
        10% odd/even balance
        10% pair/triple avoidance

    Returns (total_score, component_scores_dict).
    Component scores are each 0–1 before weighting.
    """
    split_score  = tables["split_score"]
    gap_score    = tables["gap_score"]
    decade_map   = tables["decade_map"]
    pair_penalty = tables["pair_penalty"]

    t = sorted(ticket)

    # ── A. Split-avoidance (40%) ───────────────────────────────────────
    # Average per-number split score (higher = more unpopular = better).
    s_split = sum(split_score[n] for n in t) / 6

    # ── B. Gap balance (25%) ──────────────────────────────────────────
    # Average per-number gap score; punish both over- and under-due numbers.
    s_gap = sum(gap_score[n] for n in t) / 6

    # ── C. Decade distribution (15%) ──────────────────────────────────
    # Perfect = 1 number from each of the 5 decades (or near-even spread).
    # Score = unique decades covered / 5, then bonus if well spread.
    decades_hit = len({decade_map[n] for n in t})
    # Count per decade
    dec_counts = Counter(decade_map[n] for n in t)
    # Variance penalty: low variance across decades = better distribution
    counts = list(dec_counts.values())
    # Pad with zeros for empty decades so variance reflects empty ones
    for d in range(5):
        if d not in dec_counts:
            counts.append(0)
    variance = float(np.var(counts))
    max_variance = float(np.var([6, 0, 0, 0, 0]))  # worst case
    s_decade = (decades_hit / 5) * (1.0 - variance / max_variance)

    # ── D. Odd/Even balance (10%) ─────────────────────────────────────
    # Ideal: 3 odd + 3 even. Score peaks at 3/3, falls symmetrically.
    odds = sum(1 for n in t if n % 2 != 0)
    # Distance from ideal (3): 0→1.0, 1→0.83, 2→0.5, 3→0.0
    s_odd = 1.0 - abs(odds - 3) / 3

    # ── E. Pair/triple avoidance (10%) ────────────────────────────────
    # Average penalty across all C(6,2)=15 pairs in the ticket.
    ticket_pairs = list(combinations(t, 2))
    avg_pair_penalty = sum(pair_penalty.get(p, 0.0) for p in ticket_pairs) / len(ticket_pairs)
    s_pair = 1.0 - avg_pair_penalty  # invert: low penalty = high score

    total = (
        W_SPLIT  * s_split +
        W_GAP    * s_gap   +
        W_DECADE * s_decade +
        W_ODD    * s_odd   +
        W_PAIR   * s_pair
    ) * 100  # → 0–100

    components = {
        "Split avoidance": round(s_split   * 100, 1),
        "Gap balance":     round(s_gap     * 100, 1),
        "Decade spread":   round(s_decade  * 100, 1),
        "Odd/Even":        round(s_odd     * 100, 1),
        "Pair avoidance":  round(s_pair    * 100, 1),
        "Total":           round(total, 2),
    }
    return total, components


def generate_scored_tickets(
    pool: list[int],
    tables: dict,
    excluded: set,
    recent_set: set,
    sum_min: int, sum_max: int,
    above31_min: int,
    no_consec: bool,
    no_arith: bool,
    odd_min: int, odd_max: int,
    n_candidates: int = CANDIDATES,
    top_n: int = TOP_N,
) -> list[tuple[float, list[int], dict]]:
    """
    Sample n_candidates random tickets from pool, apply hard filters,
    score the survivors, return the top_n by total score.

    Returns list of (score, ticket, components) sorted descending.
    """
    p = [n for n in pool if n not in excluded and n not in recent_set]
    if len(p) < 6:
        p = [n for n in range(1, 50) if n not in excluded]
    if len(p) < 6:
        p = list(range(1, 50))

    scored: list[tuple[float, list[int], dict]] = []
    seen_tickets: set[tuple] = set()

    for _ in range(n_candidates):
        t = sorted(random.sample(p, 6))
        key = tuple(t)
        if key in seen_tickets:
            continue
        seen_tickets.add(key)

        if not passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            continue

        score, components = score_ticket(t, tables)
        scored.append((score, t, components))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


def gen_ticket(pool, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=8000):
    """Fast random generator — used by strategy modes as the candidate source."""
    p = [int(n) for n in pool if 1 <= n <= 49]
    if len(p) < 6: p = list(range(1, 50))
    for _ in range(tries):
        t = sorted(random.sample(p, 6))
        if passes(t, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max):
            return t
    return sorted(random.sample(p, 6))


def decade_spread_ticket(excluded, recent, sum_min, sum_max, above31_min, no_consec, no_arith, odd_min, odd_max, tries=15000):
    """Generate one ticket guaranteed to span at least 4 decades."""
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
    """Generate one ticket using the delta (gap between consecutive numbers) system."""
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
st.sidebar.subheader("Hard filters")
sum_min     = st.sidebar.slider("Min sum", 60, 180, 115)
sum_max     = st.sidebar.slider("Max sum", 120, 250, 185)
above31_min = st.sidebar.slider("Min numbers above 31", 0, 6, 3,
    help="The only real edge — avoids jackpot splitting")
odd_min, odd_max = st.sidebar.select_slider(
    "Odd count range", options=list(range(7)), value=(2, 4))
no_consec = st.sidebar.checkbox("No consecutive numbers", value=True)
no_arith  = st.sidebar.checkbox("No arithmetic sequences", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Scoring weights")

# ── Presets ────────────────────────────────────────────────────────────
PRESETS = {
    "🛡️ Split Avoider":    (60, 15, 10,  5, 10),
    "⚖️ Balanced":         (40, 25, 15, 10, 10),
    "📊 Statistician":     (20, 40, 20, 10, 10),
    "🗺️ Cartographer":    (25, 15, 40, 10, 10),
    "🤝 Contrarian":       (30, 10, 15,  5, 40),
    "🎲 Pure Equal":       (20, 20, 20, 20, 20),
    "🧮 Odd/Even Purist":  (25, 20, 20, 30,  5),
}

st.sidebar.caption("Pick a preset to auto-fill weights, then fine-tune with the sliders.")

# Radio — no selection = Custom (sliders untouched)
preset_names = ["✏️ Custom"] + list(PRESETS.keys())
chosen_preset = st.sidebar.radio(
    "Strategy preset",
    preset_names,
    index=1,          # default = Balanced
    label_visibility="collapsed",
)

if chosen_preset == "✏️ Custom":
    default_split, default_gap, default_decade, default_odd, default_pair = 40, 25, 15, 10, 10
else:
    default_split, default_gap, default_decade, default_odd, default_pair = PRESETS[chosen_preset]

st.sidebar.caption("Fine-tune below (must total 100%):")
w_split  = st.sidebar.slider("↳ Split avoidance %",     0, 100, default_split,  key="w_split")
w_gap    = st.sidebar.slider("↳ Gap balance %",          0, 100, default_gap,    key="w_gap")
w_decade = st.sidebar.slider("↳ Decade distribution %",  0, 100, default_decade, key="w_decade")
w_odd    = st.sidebar.slider("↳ Odd/even balance %",     0, 100, default_odd,    key="w_odd")
w_pair   = st.sidebar.slider("↳ Pair avoidance %",       0, 100, default_pair,   key="w_pair")

weight_total = w_split + w_gap + w_decade + w_odd + w_pair
if weight_total != 100:
    st.sidebar.warning(f"Weights sum to {weight_total}% — must equal 100%")
else:
    W_SPLIT  = w_split  / 100
    W_GAP    = w_gap    / 100
    W_DECADE = w_decade / 100
    W_ODD    = w_odd    / 100
    W_PAIR   = w_pair   / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Generator options")
n_tickets = st.sidebar.slider("Top N tickets to show", 1, 10, 5)
n_candidates_k = st.sidebar.select_slider(
    "Candidates to evaluate",
    options=[10_000, 25_000, 50_000, 100_000, 200_000],
    value=100_000,
    help="More candidates = better top-10, but slower (~1–3s per 100k on Streamlit Cloud)"
)

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

with st.spinner("Loading live draw results from ca.lottonumbers.com..."):
    df = load_draws(n_years)

if df.empty:
    st.error(
        "❌ Could not load draw data from ca.lottonumbers.com. "
        "The site may be temporarily unavailable. "
        "Try clicking **Refresh live data** in the sidebar, or wait a few minutes and try again."
    )
    st.stop()

st.success(f"✅ {len(df)} draws loaded — most recent: **{df['DATE'].iloc[-1]}**")

# ── Last draw banner ──────────────────────────
last_row   = df.iloc[-1]
last_nums  = [int(last_row[c]) for c in NUMBER_COLS]
last_bonus = int(last_row["BONUS"]) if pd.notna(last_row.get("BONUS")) else None
prev_row   = df.iloc[-2] if len(df) >= 2 else last_row
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

# ── Build analytics ───────────────────────────
recent_set = set()
if exclude_recent:
    for _, row in df.tail(n_recent_exclude).iterrows():
        recent_set.update(int(row[c]) for c in NUMBER_COLS)

freq    = frequency_table(df)
gaps_df = gap_table(df)
hot6    = [n for n, _ in freq.most_common(6)]
cold6   = [n for n, _ in freq.most_common()[:-7:-1]]

full_pool  = [n for n in range(1, 50) if n not in excluded]
fresh_pool = [n for n in range(1, 50) if n not in excluded and n not in recent_set]
hot_pool   = [n for n, _ in freq.most_common(20) if n not in excluded and n not in recent_set]
cold_pool  = [n for n, _ in freq.most_common()[:-21:-1] if n not in excluded and n not in recent_set]
over15     = gaps_df.head(15)["Number"].tolist()
over_pool  = [n for n in over15 if n not in excluded and n not in recent_set]
gut_pool   = [n for n in [2,4,5,6,7,8,13,14,15,19,23,26,27,29,32,33,35,37,39,42,43,44,47,48]
              if n not in excluded and n not in recent_set]

# Build scoring tables once (cheap, reused across all tabs)
tables = build_scoring_tables(df, freq, gaps_df)

delta_dist = Counter()
for _, row in df.iterrows():
    s = sorted(row[NUMBER_COLS].tolist())
    delta_dist.update(s[i + 1] - s[i] for i in range(5))


# ──────────────────────────────────────────────
# TICKET CARD RENDERER  (must be defined before the tab block that calls it)
# ──────────────────────────────────────────────

def _render_ticket_card(ticket, score, components, recent_set, draw_num, rank=None, is_lucky=False):
    """Render a single ticket card with score breakdown bar chart."""
    in_rec = [n for n in ticket if n in recent_set]
    above  = sum(1 for n in ticket if n > 31)
    odds   = sum(1 for n in ticket if n % 2 != 0)
    s      = sum(ticket)

    label = f"🩷 Draw {draw_num} — Lucky combo" if is_lucky else f"#{rank} — Draw {draw_num}"
    balls = "  ".join([f"`{n}`" for n in sorted(ticket)])
    st.markdown(f"**{label}:** {balls}  — **Score: {score:.1f}/100**")

    comp_labels = list(components.keys())[:-1]  # exclude Total
    comp_values = [components[k] for k in comp_labels]

    c_balls, c_chart = st.columns([2, 3])

    with c_balls:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sum", s)
        m2.metric("Above 31", f"{above}/6")
        m3.metric("Odd/Even", f"{odds}/{6 - odds}")
        m4, m5 = st.columns(2)
        m4.metric("Split risk", split_risk(ticket).split()[0])
        m5.metric("Score", f"{score:.1f}/100")
        if in_rec:
            st.warning(f"⚠️ {in_rec} in last draw(s)")

    with c_chart:
        fig = go.Figure(go.Bar(
            x=comp_values,
            y=comp_labels,
            orientation="h",
            marker_color=["#4C78A8", "#54A24B", "#E45756", "#F58518", "#72B7B2"],
            text=[f"{v:.0f}" for v in comp_values],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 115], title="Component score (0–100)"),
            height=180,
            margin=dict(l=0, r=30, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{draw_num}_{score:.2f}")

    st.markdown("")  # spacer


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

    if weight_total != 100:
        st.error(f"⚠️ Sidebar weights sum to {weight_total}% — fix them to 100% before generating.")
        st.stop()

    if exclude_recent and recent_set:
        st.info(f"Auto-excluding (last {n_recent_exclude} draw{'s' if n_recent_exclude > 1 else ''}): `{sorted(recent_set)}`")

    # ── Mode selector ──────────────────────────────────────────────────
    gen_mode = st.selectbox(
        "Candidate pool strategy",
        [
            "🎯 Full pool (all numbers)",
            "📅 Decade spread",
            "🔥 Hot numbers biased",
            "❄️ Cold numbers biased",
            "⏳ Overdue numbers biased",
            "📐 Delta system",
            "🤫 Gut pick (unpopular territory)",
        ],
        help=(
            "Defines which numbers are sampled as candidates. "
            "The scorer then ranks all candidates that pass the hard filters. "
            "Decade spread and Delta pre-seed the pool with structural diversity."
        ),
    )

    st.markdown(
        f"Samples **{n_candidates_k:,} candidates** from the selected pool, "
        f"scores each on 5 dimensions, returns the **top {n_tickets}** tickets."
    )

    # Active weights display
    weight_cols = st.columns(5)
    wlabels = ["Split avoid", "Gap balance", "Decade spread", "Odd/Even", "Pair avoid"]
    wpcts   = [w_split, w_gap, w_decade, w_odd, w_pair]
    for col, lbl, pct in zip(weight_cols, wlabels, wpcts):
        col.metric(lbl, f"{pct}%")

    show_lucky = st.checkbox("Score & show my lucky combo first", value=True)

    if st.button("🎲 Run scorer", type="primary"):

        # ── Resolve candidate pool from strategy ───────────────────────
        if "Decade spread" in gen_mode:
            # Pre-generate a large pool of decade-spread candidates, then score them
            pre_pool = []
            for _ in range(n_candidates_k):
                t = decade_spread_ticket(
                    excluded, recent_set,
                    sum_min, sum_max, above31_min,
                    no_consec, no_arith, odd_min, odd_max,
                    tries=50,
                )
                if t:
                    pre_pool.extend(t)
            # Flatten back to a unique number pool weighted by appearance
            candidate_pool = list(set(pre_pool)) if pre_pool else fresh_pool or full_pool

        elif "Delta" in gen_mode:
            pre_pool = []
            for _ in range(n_candidates_k):
                t = delta_ticket(
                    delta_dist, excluded, recent_set,
                    sum_min, sum_max, above31_min,
                    no_consec, no_arith, odd_min, odd_max,
                    tries=20,
                )
                if t:
                    pre_pool.extend(t)
            candidate_pool = list(set(pre_pool)) if pre_pool else fresh_pool or full_pool

        elif "Hot" in gen_mode:
            candidate_pool = hot_pool or fresh_pool or full_pool

        elif "Cold" in gen_mode:
            candidate_pool = cold_pool or fresh_pool or full_pool

        elif "Overdue" in gen_mode:
            candidate_pool = (over_pool + fresh_pool) or full_pool

        elif "Gut" in gen_mode:
            candidate_pool = gut_pool or fresh_pool or full_pool

        else:  # Full pool
            candidate_pool = fresh_pool or full_pool

        with st.spinner(f"Scoring {n_candidates_k:,} candidates from {gen_mode.split('(')[0].strip()} pool…"):
            top_tickets = generate_scored_tickets(
                pool=candidate_pool,
                tables=tables,
                excluded=excluded,
                recent_set=recent_set,
                sum_min=sum_min, sum_max=sum_max,
                above31_min=above31_min,
                no_consec=no_consec, no_arith=no_arith,
                odd_min=odd_min, odd_max=odd_max,
                n_candidates=n_candidates_k,
                top_n=n_tickets,
            )

        if not top_tickets:
            st.error("No tickets passed the hard filters. Try relaxing the sidebar filters or switching strategy.")
            st.stop()

        draw_num = 1

        # ── Lucky combo ────────────────────────────────────────────────
        if show_lucky and lucky_str.strip():
            try:
                lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
                if len(lucky) == 6:
                    lucky_score, lucky_comp = score_ticket(lucky, tables)
                    st.markdown("---")
                    st.markdown(f"### 🩷 Your lucky combo — Score: **{lucky_score:.1f} / 100**")
                    _render_ticket_card(lucky, lucky_score, lucky_comp, recent_set, draw_num, is_lucky=True)
                    draw_num += 1
            except Exception:
                pass

        # ── Top scored tickets ─────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### 🏆 Top {len(top_tickets)} from {n_candidates_k:,} candidates · {gen_mode.split('(')[0].strip()} pool")

        for rank, (score, ticket, components) in enumerate(top_tickets, 1):
            _render_ticket_card(ticket, score, components, recent_set, draw_num, rank=rank)
            draw_num += 1

        # ── Score distribution chart ───────────────────────────────────
        st.markdown("---")
        st.markdown("#### Score distribution across all passing candidates")
        st.caption("Top tickets sit in the right tail. The cutoff line shows where top-N begins.")
        with st.spinner("Sampling distribution…"):
            sample_scored = generate_scored_tickets(
                pool=candidate_pool,
                tables=tables,
                excluded=excluded,
                recent_set=recent_set,
                sum_min=sum_min, sum_max=sum_max,
                above31_min=above31_min,
                no_consec=no_consec, no_arith=no_arith,
                odd_min=odd_min, odd_max=odd_max,
                n_candidates=min(n_candidates_k, 10_000),
                top_n=min(n_candidates_k, 10_000),
            )
        if sample_scored:
            all_scores = [s for s, _, _ in sample_scored]
            fig_dist = px.histogram(
                x=all_scores, nbins=40,
                labels={"x": "Score", "y": "Count"},
                title=f"Score distribution ({len(all_scores):,} passing tickets sampled)",
                color_discrete_sequence=["#4C78A8"],
            )
            if top_tickets:
                min_top = min(s for s, _, _ in top_tickets)
                fig_dist.add_vline(
                    x=min_top, line_dash="dash", line_color="orange",
                    annotation_text=f"Top {n_tickets} cutoff ({min_top:.1f})",
                    annotation_position="top right",
                )
            st.plotly_chart(fig_dist, use_container_width=True)


def _render_ticket_card(ticket, score, components, recent_set, draw_num, rank=None, is_lucky=False):
    """Render a single ticket card with score breakdown."""
    in_rec = [n for n in ticket if n in recent_set]
    above  = sum(1 for n in ticket if n > 31)
    odds   = sum(1 for n in ticket if n % 2 != 0)
    s      = sum(ticket)

    label = f"🩷 Draw {draw_num} — Lucky combo" if is_lucky else f"#{rank} — Draw {draw_num}"
    balls = "  ".join([f"`{n}`" for n in sorted(ticket)])
    st.markdown(f"**{label}:** {balls}  — **Score: {score:.1f}/100**")

    # Score radar / bar
    comp_labels = list(components.keys())[:-1]  # exclude Total
    comp_values = [components[k] for k in comp_labels]

    c_balls, c_chart = st.columns([2, 3])

    with c_balls:
        m1, m2, m3 = st.columns(3)
        m1.metric("Sum", s)
        m2.metric("Above 31", f"{above}/6")
        m3.metric("Odd/Even", f"{odds}/{6 - odds}")
        m4, m5 = st.columns(2)
        m4.metric("Split risk", split_risk(ticket).split()[0])
        m5.metric("Score", f"{score:.1f}/100")
        if in_rec:
            st.warning(f"⚠️ {in_rec} in last draw(s)")

    with c_chart:
        fig = go.Figure(go.Bar(
            x=comp_values,
            y=comp_labels,
            orientation="h",
            marker_color=["#4C78A8", "#54A24B", "#E45756", "#F58518", "#72B7B2"],
            text=[f"{v:.0f}" for v in comp_values],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 105], title="Component score (0–100)"),
            height=180,
            margin=dict(l=0, r=30, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{draw_num}_{score:.2f}")

    st.markdown("")  # spacer


# ── LUCKY COMBO ───────────────────────────────
with tab_lucky:
    st.subheader("🩷 Lucky combo deep audit")
    try:
        lucky = sorted(int(x.strip()) for x in lucky_str.split(",") if x.strip().isdigit())
        if len(lucky) == 6:
            lucky_score, lucky_comp = score_ticket(lucky, tables)
            s = sum(lucky); above = sum(1 for n in lucky if n > 31); odds = sum(1 for n in lucky if n % 2 != 0)
            in_rec = [n for n in lucky if n in recent_set]

            st.markdown(f"### Overall score: **{lucky_score:.1f} / 100**")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Sum", s, delta="OK ✓" if 115 <= s <= 185 else "⚠️ Out of range")
            c2.metric("Above 31", f"{above}/6", delta="OK ✓" if above >= 3 else "⚠️ Too low")
            c3.metric("Odd/Even", f"{odds}/{6 - odds}")
            c4.metric("Consecutive?", "Yes ⚠️" if has_consec(lucky) else "No ✓")
            c5.metric("Split risk", split_risk(lucky).split()[0])

            if in_rec: st.warning(f"⚠️ {in_rec} appeared in the last {n_recent_exclude} draw(s).")

            # Score breakdown bar
            comp_labels = list(lucky_comp.keys())[:-1]
            comp_values = [lucky_comp[k] for k in comp_labels]
            weights_pct  = [w_split, w_gap, w_decade, w_odd, w_pair]
            contrib      = [v * w / 100 for v, w in zip(comp_values, weights_pct)]

            fig_lucky = go.Figure()
            fig_lucky.add_trace(go.Bar(
                name="Component score",
                x=comp_labels, y=comp_values,
                marker_color="#4C78A8",
                text=[f"{v:.0f}" for v in comp_values],
                textposition="outside",
            ))
            fig_lucky.add_trace(go.Bar(
                name="Weighted contribution",
                x=comp_labels, y=[c * 100 / (w/100) * (w/100) for c, w in zip(contrib, weights_pct)],
                marker_color="#F58518",
                opacity=0.5,
            ))
            fig_lucky.update_layout(
                barmode="overlay",
                yaxis=dict(range=[0, 115], title="Score (0–100)"),
                height=300,
                margin=dict(t=10),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_lucky, use_container_width=True)

            ctx = pd.DataFrame({
                "Number": lucky,
                "All-time frequency": [freq.get(n, 0) for n in lucky],
                "Draws since last seen": [int(gaps_df.set_index("Number").loc[n, "Gap"]) for n in lucky],
                "Popularity score": [popularity_score(n) for n in lucky],
                "Split score": [round(tables["split_score"][n] * 100, 1) for n in lucky],
                "Gap score": [round(tables["gap_score"][n] * 100, 1) for n in lucky],
                "Above 31": ["Yes" if n > 31 else "No" for n in lucky],
                "In recent draw?": ["🔴 Yes" if n in recent_set else "No" for n in lucky],
            })
            st.dataframe(ctx, use_container_width=True)

            if s < 115: st.error(f"Sum {s} is below the 115–185 historical range.")
            if above == 0: st.error("Zero numbers above 31 — very high split risk.")
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
    c1.metric("🔥 Top 6 hot", str(hot6))
    c2.metric("❄️ Bottom 6 cold", str(cold6))

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
                  color="Count", color_continuous_scale="Viridis",
                  title="Top 25 most co-occurring pairs")
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
                past  = [tuple(sorted(int(row[c]) for c in NUMBER_COLS)) for _, row in df.iterrows()]
                count = past.count(check)
                if count > 0:
                    st.success(f"✅ Appeared {count} time(s) in {len(df)} draws.")
                else:
                    st.info(f"❌ Never appeared in {len(df)} draws analyzed.")
                # Also score this combo
                score, components = score_ticket(list(check), tables)
                st.markdown(f"**Score if you played this:** {score:.1f} / 100")
                comp_df = pd.DataFrame([
                    {"Dimension": k, "Score": v}
                    for k, v in components.items() if k != "Total"
                ])
                st.dataframe(comp_df, use_container_width=True)
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
    "Data: ca.lottonumbers.com · Cache refreshes every hour."
)
