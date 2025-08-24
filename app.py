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
    layout="wide"
)

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and backtest strategies.")

# ======================
# Helper Functions
# ======================

def extract_numbers_and_bonus(df):
    """Extract 6 numbers and bonus from historical draw dataframe."""
    numbers_cols = [f"Number{i}" for i in range(1, 7)]
    df[numbers_cols] = df[numbers_cols].astype(int)
    df["Bonus"] = df["Bonus"].astype(int)
    return df

def compute_frequencies(df):
    """Compute frequency of each number."""
    numbers = df[[f"Number{i}" for i in range(1, 7)]].values.flatten()
    return Counter(numbers)

def compute_pair_frequencies(df):
    """Compute frequency of pairs."""
    pairs = []
    for row in df[[f"Number{i}" for i in range(1, 7)]].values:
        for combo in combinations(sorted(row), 2):
            pairs.append(combo)
    return Counter(pairs)

def compute_delta_distribution(df):
    """Compute distribution of differences between consecutive numbers."""
    deltas = []
    for row in df[[f"Number{i}" for i in range(1, 7)]].values:
        row = sorted(row)
        deltas.extend([row[i+1]-row[i] for i in range(5)])
    return Counter(deltas)

# ======================
# Ticket Generators
# ======================

def generate_random_ticket():
    return sorted(random.sample(range(1, 50), 6))

def generate_delta_ticket(delta_counter):
    deltas = [d for d,_ in delta_counter.most_common(10)]
    ticket = [random.randint(1, 44)]
    while len(ticket) < 6:
        next_num = ticket[-1] + random.choice(deltas)
        if next_num <= 49 and next_num not in ticket:
            ticket.append(next_num)
    return sorted(ticket)

def generate_balanced_ticket():
    """Generate ticket balancing low/high numbers and odd/even."""
    low = list(range(1, 25))
    high = list(range(25, 50))
    ticket = random.sample(low, 3) + random.sample(high, 3)
    return sorted(ticket)

def generate_zone_ticket():
    """Generate ticket by picking one number from each dozen zone."""
    zones = [range(1, 10), range(10, 20), range(20, 30), range(30, 40), range(40, 50)]
    ticket = [random.choice(zone) for zone in zones[:5]]
    ticket.append(random.randint(1, 49))
    return sorted(ticket)

# ======================
# Sidebar Controls
# ======================

st.sidebar.header("Upload Historical Draws")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.header("Ticket Generator Settings")
num_tickets = st.sidebar.slider("Number of Tickets", 1, 20, 5)

# ======================
# Main App Logic
# ======================

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = extract_numbers_and_bonus(df)
    
    st.subheader("Historical Numbers Frequency")
    freq = compute_frequencies(df)
    freq_df = pd.DataFrame(freq.items(), columns=["Number", "Frequency"]).sort_values("Number")
    st.dataframe(freq_df)
    
    fig = px.bar(freq_df, x="Number", y="Frequency", title="Number Frequency")
    st.plotly_chart(fig)
    
    st.subheader("Pair Frequencies")
    pair_freq = compute_pair_frequencies(df)
    top_pairs = pd.DataFrame(pair_freq.most_common(20), columns=["Pair", "Count"])
    st.dataframe(top_pairs)
    
    st.subheader("Delta Distribution")
    delta_freq = compute_delta_distribution(df)
    delta_df = pd.DataFrame(delta_freq.items(), columns=["Delta", "Count"]).sort_values("Delta")
    st.dataframe(delta_df)
    
    fig2 = px.bar(delta_df, x="Delta", y="Count", title="Delta Distribution")
    st.plotly_chart(fig2)
    
    st.subheader("Generate Tickets")
    generator = st.selectbox("Choose Generator", ["Random", "Delta", "Balanced", "Zone"])
    
    tickets = []
    for _ in range(num_tickets):
        if generator == "Random":
            tickets.append(generate_random_ticket())
        elif generator == "Delta":
            tickets.append(generate_delta_ticket(delta_freq))
        elif generator == "Balanced":
            tickets.append(generate_balanced_ticket())
        elif generator == "Zone":
            tickets.append(generate_zone_ticket())
    
    st.write(tickets)
else:
    st.info("Please upload a CSV file with historical Lotto 6/49 draws.")
