import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import random

# ---------------------------
# Streamlit App Config
# ---------------------------
st.set_page_config(page_title="Canada Lotto 6/49 Analyzer", layout="wide")
st.title("🍁 Canada Lotto 6/49 Analyzer")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload your Lotto 6/49 CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Detect date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break

    if date_col:
        # Clean ordinal suffixes (1st, 2nd, 3rd...)
        df[date_col] = df[date_col].astype(str).apply(lambda x: re.sub(r"(\d+)(st|nd|rd|th)", r"\1", x))
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Detect drawn numbers
    number_cols = [c for c in df.columns if "number" in c.lower()]

    # ---------------------------
    # Tabs
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Data Preview", "🔥 Frequency", "🌡️ Hot/Cold/Overdue", "🎲 Simulation", "🤖 ML Prediction"]
    )

    # ---------------------------
    # Tab 1: Data Preview
    # ---------------------------
    with tab1:
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        display_df = df.copy()
        if date_col:
            dates = df[date_col]
            display_df["DATE"] = pd.Series(dates.values, index=display_df.index)

        st.subheader("📅 Processed Data")
        st.dataframe(display_df.head())

    # ---------------------------
    # Tab 2: Frequency Analysis
    # ---------------------------
    with tab2:
        all_numbers = df[number_cols].values.flatten()
        all_numbers = all_numbers[~pd.isna(all_numbers)]
        all_numbers = all_numbers.astype(int)

        freq = pd.Series(all_numbers).value_counts().sort_index()
        freq_df = freq.reset_index()
        freq_df.columns = ["Number", "Frequency"]

        st.subheader("🔥 Number Frequency")
        fig_freq = px.bar(freq_df, x="Number", y="Frequency", title="Frequency of Each Number")
        st.plotly_chart(fig_freq, use_container_width=True)

    # ---------------------------
    # Tab 3: Hot / Cold / Overdue
    # ---------------------------
    with tab3:
        freq_df = pd.Series(all_numbers).value_counts().sort_index().reset_index()
        freq_df.columns = ["Number", "Frequency"]

        hot_numbers = freq_df.sort_values("Frequency", ascending=False).head(6)
        cold_numbers = freq_df.sort_values("Frequency", ascending=True).head(6)

        col1, col2 = st.columns(2)
        with col1:
            st.write("🔥 **Hot Numbers**")
            st.table(hot_numbers)
        with col2:
            st.write("❄️ **Cold Numbers**")
            st.table(cold_numbers)

        if date_col:
            last_draw_date = df[date_col].max()
            last_draw_numbers = df[df[date_col] == last_draw_date][number_cols].values.flatten()
            overdue = [n for n in range(1, 50) if n not in last_draw_numbers]
            st.subheader("⏳ Overdue Numbers")
            st.write(overdue)

    # ---------------------------
    # Tab 4: Simulation
    # ---------------------------
    with tab4:
        st.subheader("🎲 Lottery Simulation")
        num_sim = st.number_input("Number of simulations:", min_value=100, max_value=100000, value=1000, step=100)

        def simulate_draw():
            return sorted(random.sample(range(1, 50), 6))

        sims = [simulate_draw() for _ in range(num_sim)]
        sim_freq = pd.Series([n for draw in sims for n in draw]).value_counts().sort_index()

        fig_sim = px.bar(
            x=sim_freq.index,
            y=sim_freq.values,
            labels={"x": "Number", "y": "Simulated Frequency"},
            title=f"Simulation of {num_sim} Lotto 6/49 Draws"
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    # ---------------------------
    # Tab 5: Machine Learning Prediction
    # ---------------------------
    with tab5:
        st.subheader("🤖 Machine Learning Prediction (Experimental)")

        if len(df) > 20:  # need enough history
            # Prepare features/labels
            df = df.sort_values(by=date_col) if date_col else df.reset_index(drop=True)
            X = pd.DataFrame({"DrawIndex": range(len(df))})
            Y = pd.DataFrame(0, index=df.index, columns=range(1, 50))
            for i, row in df[number_cols].iterrows():
                for n in row.values:
                    if 1 <= n <= 49:
                        Y.at[i, n] = 1

            model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
            model.fit(X, Y)

            # Predict next draw
            next_X = pd.DataFrame({"DrawIndex": [len(df)]})
            pred_probs = model.predict_proba(next_X)

            probs = {n: pred_probs[i][0][1] for i, n in enumerate(range(1, 50))}
            top_pred = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:6]

            st.write("Top predicted numbers for the next draw:")
            for n, p in top_pred:
                st.write(f"Number {n}: {p:.2%} chance")

            fig_pred = px.bar(
                x=[n for n, _ in top_pred],
                y=[p for _, p in top_pred],
                labels={"x": "Number", "y": "Predicted Probability"},
                title="Top Predicted Numbers"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("Not enough historical draws for ML prediction. Upload more data.")
            # ---------------------------
    # Tab 6: Strategy Tools
    # ---------------------------
    tab6 = st.tabs(["🎯 Strategy Tools"])[0]

    with tab6:
        st.subheader("🎯 Strategy Tools")

        # Cluster / Zone Coverage
        st.markdown("### 📌 Cluster / Zone Coverage")
        zones = [(1,10),(11,20),(21,30),(31,40),(41,49)]
        zone_counts = {f"{a}-{b}":0 for a,b in zones}
        for draw in df[number_cols].values:
            for num in draw:
                for a,b in zones:
                    if a <= num <= b:
                        zone_counts[f"{a}-{b}"] += 1
                        break
        st.bar_chart(pd.Series(zone_counts))

        # Delta System
        st.markdown("### 🔀 Delta System")
        deltas_input = st.text_input("Enter deltas (comma separated)", "5,8,7,3,12,14")
        try:
            deltas = list(map(int, deltas_input.split(",")))
            numbers = []
            total = 0
            for d in deltas:
                total += d
                if total <= 49:
                    numbers.append(total)
            st.write("Generated numbers:", numbers)
        except:
            st.warning("Invalid delta input")

        # Sum & Spread Filters
        st.markdown("### ➕ Sum & Spread Filters")
        num_pick = sorted(random.sample(range(1,50),6))
        total_sum = sum(num_pick)
        spread = max(num_pick) - min(num_pick)

        st.write("Random pick:", num_pick)
        st.write("Sum:", total_sum, " Spread:", spread)

        min_sum, max_sum = st.slider("Acceptable sum range", 50, 300, (100,180))
        min_spread, max_spread = st.slider("Acceptable spread range", 1, 48, (15,35))

        if min_sum <= total_sum <= max_sum and min_spread <= spread <= max_spread:
            st.success("✅ This pick meets your filters!")
        else:
            st.error("❌ This pick does not meet your filters.")
