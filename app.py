import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Canada Lotto AI", layout="wide")

# ------------------------
# Load Data (flexible)
# ------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("649.csv")  # local copy
    except FileNotFoundError:
        st.warning("⚠️ No local '649.csv' found. Please upload one below.")
        uploaded = st.file_uploader("Upload your Lotto 6/49 CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df

df = load_data()
number_cols = [c for c in df.columns if c.startswith("N")]

# ------------------------
# Helper: Clean numbers safely
# ------------------------
def get_all_numbers():
    if df.empty or not number_cols:
        return pd.Series([], dtype=int)
    all_numbers = df[number_cols].values.flatten()
    all_numbers = pd.to_numeric(all_numbers, errors="coerce")
    all_numbers = pd.Series(all_numbers).dropna().astype(int)
    return all_numbers

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs([
    "Data Explorer", "Frequency Analysis", "Hot & Cold Numbers",
    "Simulation", "Machine Learning", "Strategy Tools"
])

# ------------------------
# Tab 1: Data Explorer
# ------------------------
with tabs[0]:
    st.header("📊 Lotto Data Explorer")
    if df.empty:
        st.info("Please upload a CSV file to explore data.")
    else:
        st.dataframe(df.head(50))

# ------------------------
# Tab 2: Frequency Analysis
# ------------------------
with tabs[1]:
    st.header("🔢 Frequency Analysis")
    all_numbers = get_all_numbers()
    if all_numbers.empty:
        st.info("No numbers available for analysis.")
    else:
        freq = Counter(all_numbers)
        freq_df = pd.DataFrame(sorted(freq.items()), columns=["Number", "Frequency"])
        st.bar_chart(freq_df.set_index("Number"))
        st.dataframe(freq_df)

# ------------------------
# Tab 3: Hot & Cold Numbers
# ------------------------
with tabs[2]:
    st.header("🔥 Hot & ❄️ Cold Numbers")
    all_numbers = get_all_numbers()
    if all_numbers.empty:
        st.info("No numbers available for analysis.")
    else:
        freq = Counter(all_numbers)
        hot = sorted(freq.items(), key=lambda x: -x[1])[:10]
        cold = sorted(freq.items(), key=lambda x: x[1])[:10]
        st.subheader("Top 10 Hot Numbers")
        st.table(hot)
        st.subheader("Top 10 Cold Numbers")
        st.table(cold)

# ------------------------
# Tab 4: Simulation
# ------------------------
with tabs[3]:
    st.header("🎲 Simulation")
    if st.button("Generate Random Ticket"):
        ticket = sorted(random.sample(range(1, 50), 6))
        st.success(f"Your ticket: {ticket}")

# ------------------------
# Tab 5: Machine Learning
# ------------------------
with tabs[4]:
    st.header("🤖 Machine Learning")
    if df.empty or not number_cols:
        st.info("Upload data to enable machine learning.")
    else:
        X = df[number_cols].dropna().astype(int)
        y = (X.max(axis=1) % 2 == 0).astype(int)  # dummy target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

# ------------------------
# Tab 6: Strategy Tools
# ------------------------
with tabs[5]:
    st.header("🛠 Strategy Tools")

    if df.empty or not number_cols:
        st.info("Upload data to use strategy tools.")
    else:
        # Cluster / Zone Coverage
        st.subheader("Cluster / Zone Coverage")
        cluster_ranges = [(1,10),(11,20),(21,30),(31,40),(41,49)]
        last_draw = df[number_cols].iloc[-1].dropna().astype(int).tolist()
        coverage = {f"{a}-{b}": sum(1 for n in last_draw if a <= n <= b) for a,b in cluster_ranges}
        st.write("Last draw coverage:", coverage)

        # Delta System
        st.subheader("Delta System")
        if st.button("Generate Delta Ticket"):
            deltas = sorted(random.sample(range(1,10), 6))
            ticket = [deltas[0]]
            for d in deltas[1:]:
                ticket.append(ticket[-1] + d)
            st.success(f"Delta Ticket: {ticket}")

        # Sum & Spread Filters
        st.subheader("Sum & Spread Filters")
        sum_min, sum_max = st.slider("Select sum range", 60, 200, (100,150))
        spread_max = st.slider("Max spread (difference between highest & lowest)", 10, 40, 25)

        if st.button("Generate Filtered Ticket"):
            tries = 0
            found = None
            while tries < 10000 and found is None:
                candidate = sorted(random.sample(range(1,50), 6))
                s = sum(candidate)
                spread = candidate[-1] - candidate[0]
                if sum_min <= s <= sum_max and spread <= spread_max:
                    found = candidate
                tries += 1
            if found:
                st.success(f"Filtered Ticket: {found}")
            else:
                st.error("No ticket found with these filters.")
