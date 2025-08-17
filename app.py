import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
import plotly.express as px

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer",
                   page_icon="🎲", layout="wide")
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyze historical draws, identify patterns, generate tickets, and see predictions.")

# ----------------------------
# Helper Functions
# ----------------------------
def extract_numbers_and_bonus(df):
    main_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    bonus_col = "BONUS NUMBER"
    
    if not all(col in df.columns for col in main_cols):
        return None, None, None
    
    numbers_df = df[main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None
    
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None
    
    date_col = next((c for c in ['DATE','Draw Date','Draw_Date','Date'] if c in df.columns), None)
    dates = None
    if date_col:
        import re
        df[date_col] = df[date_col].apply(lambda x: re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', str(x)) if pd.notna(x) else x)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        dates = df[date_col]
    
    return numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

@st.cache_data
def compute_frequencies(numbers_df):
    return Counter(numbers_df.values.flatten())

@st.cache_data
def compute_pair_frequencies(numbers_df, limit=500):
    pair_counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        pair_counts.update(combinations(sorted(row.values), 2))
    return pair_counts

@st.cache_data
def compute_triplet_frequencies(numbers_df, limit=500):
    triplet_counts = Counter()
    for _, row in numbers_df.tail(limit).iterrows():
        triplet_counts.update(combinations(sorted(row.values), 3))
    return triplet_counts

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {n: -1 for n in range(1, 50)}
    gaps = {n: None for n in range(1, 50)}
    
    if dates is not None:
        numbers_df = numbers_df.iloc[dates.argsort()].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)
    
    for idx, row in numbers_df.iterrows():
        for n in row.values:
            last_seen[n] = idx
    
    total_draws = len(numbers_df)
    for n in range(1, 50):
        gaps[n] = total_draws - 1 - last_seen[n] if last_seen[n] != -1 else total_draws
    return gaps

def most_common_per_draw_position(numbers_df):
    result = {}
    for col in numbers_df.columns:
        most_num, freq = Counter(numbers_df[col]).most_common(1)[0]
        result[col] = (int(most_num), freq)
    return result

def generate_ticket(pool):
    return sorted(random.sample(pool, 6))

# ----------------------------
# Main App
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to 6, BONUS NUMBER, and optional DATE"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Sort by date if available
        date_col = next((c for c in ['DATE','Draw Date','Draw_Date','Date'] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.sort_values(by=date_col, ascending=True)
        
        st.subheader(f"Uploaded Data ({len(df)} draws)")
        st.dataframe(df.reset_index(drop=True))
        
        # Slider for analysis
        draws_to_use = st.slider("Number of recent draws to analyze", 50, len(df), min(300, len(df)), step=10)
        df_recent = df.tail(draws_to_use).reset_index(drop=True)
        
        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df_recent)
        if numbers_df is None:
            st.error("CSV must have valid columns NUMBER DRAWN 1-6 with values 1-49.")
            st.stop()
        
        # Frequencies & stats
        counter = compute_frequencies(numbers_df)
        hot = [n for n,_ in counter.most_common(6)]
        cold = [n for n,_ in counter.most_common()[:-7:-1]]
        gaps = compute_number_gaps(numbers_df, dates)
        position_common = most_common_per_draw_position(numbers_df)
        
        # Hot & Cold
        st.subheader("Hot Numbers")
        st.write(", ".join(map(str, hot)))
        st.subheader("Cold Numbers")
        st.write(", ".join(map(str, cold)))
        
        # Position common
        st.subheader("Most Common Numbers by Position")
        for pos, (num,freq) in position_common.items():
            st.write(f"{pos}: {num} (appeared {freq} times)")
        
        # Frequency Chart
        freq_df = pd.DataFrame({"Number": range(1,50)})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter.get(x,0))
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency",
                     color_continuous_scale="Blues", title=f"Number Frequency (last {draws_to_use} draws)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Pair Frequency
        st.subheader(f"Number Pair Frequency (last {draws_to_use} draws)")
        pair_counts = compute_pair_frequencies(numbers_df, draws_to_use)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair","Count"]).sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count", color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)
        
        # Triplet Frequency
        st.subheader(f"Number Triplet Frequency (last {draws_to_use} draws)")
        triplet_counts = compute_triplet_frequencies(numbers_df, draws_to_use)
        triplets_df = pd.DataFrame(triplet_counts.items(), columns=["Triplet","Count"]).sort_values(by="Count", ascending=False).head(20)
        triplets_df["Triplet"] = triplets_df["Triplet"].apply(lambda x: f"{x[0]} & {x[1]} & {x[2]}")
        fig_triplets = px.bar(triplets_df, y="Triplet", x="Count", orientation='h', color="Count", color_continuous_scale="Cividis")
        fig_triplets.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_triplets, use_container_width=True)
        
        # Gap Analysis
        st.subheader("Number Gap Analysis")
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values(by="Gap", ascending=False)
        threshold = st.slider("Gap threshold for overdue numbers", 0, 100, 27)
        st.dataframe(gaps_df[gaps_df["Gap"] >= threshold])
        
        # Ticket Generator
        st.subheader("🎟️ Generate Lotto Tickets")
        strategy = st.selectbox("Ticket Generation Strategy", ["Pure Random","Bias: Hot","Bias: Cold","Bias: Overdue","Mixed"])
        num_tickets = st.slider("How many tickets?", 1, 10, 5)
        
        generated_tickets = []
        for _ in range(num_tickets):
            if strategy=="Pure Random":
                pool = list(range(1,50))
            elif strategy=="Bias: Hot":
                pool = hot + random.sample([n for n in range(1,50) if n not in hot], 43)
            elif strategy=="Bias: Cold":
                pool = cold + random.sample([n for n in range(1,50) if n not in cold], 43)
            elif strategy=="Bias: Overdue":
                top_gap = [n for n,_ in sorted(gaps.items(), key=lambda x:x[1], reverse=True)[:10]]
                pool = top_gap + random.sample([n for n in range(1,50) if n not in top_gap], 39)
            elif strategy=="Mixed":
                pool = hot[:3] + cold[:2]
                pool += random.sample([n for n in range(1,50) if n not in pool], 49-len(pool))
            generated_tickets.append(generate_ticket(pool))
        
        st.write("🎰 Generated Tickets:")
        for idx,ticket in enumerate(generated_tickets,1):
            st.write(f"Ticket {idx}: {ticket}")
        
        # ----------------------------
        # 🔍 Check if a 6-number combination appeared in full history
        # ----------------------------
        st.subheader("🔍 Check if a Draw Combination Has Already Appeared (All History)")
        
        user_draw = st.text_input("Enter 6 numbers separated by commas (e.g., 5,12,19,23,34,45):", key="check_draw_combo")
        
        if user_draw.strip():
            try:
                numbers_entered = [int(x.strip()) for x in user_draw.split(",")]
                if len(numbers_entered)!=6:
                    raise ValueError("Please enter exactly 6 numbers.")
                if not all(1 <= n <= 49 for n in numbers_entered):
                    raise ValueError("Numbers must be between 1 and 49.")
                
                user_numbers = tuple(sorted(numbers_entered))
                
                numbers_all_df = numbers_df  # if you have a full CSV, replace with full df
                dates_all = dates  # replace with full dates series if available
                
                past_draws_all = [tuple(sorted(row)) for row in numbers_all_df.values.tolist()]
                matches_idx = [i for i,row in enumerate(numbers_all_df.values.tolist()) if tuple(sorted(row))==user_numbers]
                
                if matches_idx:
                    st.success(f"✅ This combination appeared {len(matches_idx)} time(s) in history!")
                    if dates_all is not None:
                        st.write("Occurrences:")
                        for i in matches_idx:
                            clean_numbers = [int(x) for x in numbers_all_df.iloc[i].values]
                            st.write(f"- {pd.to_datetime(dates_all.iloc[i]).date()}: {clean_numbers}")
                    else:
                        st.write("Occurrences (row indexes):")
                        st.write([i+1 for i in matches_idx])
                else:
                    st.error("❌ This combination has never appeared in history.")
                
            except Exception as e:
                st.error(f"⚠️ Invalid input: {e}")
        
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

else:
    st.info("Please upload a CSV file with Lotto 6/49 draw results.")
