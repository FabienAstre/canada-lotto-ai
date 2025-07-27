# ... (all previous code unchanged, just move the predictive model block down)

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload a Lotto 6/49 CSV file",
    type=["csv"],
    help="CSV with columns: NUMBER DRAWN 1 to NUMBER DRAWN 6 and BONUS NUMBER",
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data (Last 30 draws):")
        st.dataframe(df.tail(30))

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)
        if numbers_df is None:
            st.error("CSV must have columns 'NUMBER DRAWN 1' to 'NUMBER DRAWN 6' with values 1-49.")
            st.stop()

        # --- Frequency Analysis ---
        counter = compute_frequencies(numbers_df)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num, _ in counter.most_common()[:-7:-1]]

        st.subheader("Hot Numbers:")
        st.write(", ".join(map(str, hot)))
        st.subheader("Cold Numbers:")
        st.write(", ".join(map(str, cold)))

        # Frequency Chart
        freq_df = pd.DataFrame({"Number": list(range(1, 50))})
        freq_df["Frequency"] = freq_df["Number"].apply(lambda x: counter[x] if x in counter else 0)
        fig = px.bar(freq_df, x="Number", y="Frequency", color="Frequency", title="Number Frequency", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        # Pair frequency chart (last 500 draws)
        st.subheader("Number Pair Frequency (last 500 draws)")
        pair_counts = compute_pair_frequencies(numbers_df, limit=500)
        pairs_df = pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"]).sort_values(by="Count", ascending=False).head(20)
        pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")
        fig_pairs = px.bar(pairs_df, y="Pair", x="Count", orientation='h', color="Count", color_continuous_scale="Viridis")
        fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pairs, use_container_width=True)

        # Gap Analysis
        st.subheader("Number Gap Analysis")
        gaps = compute_number_gaps(numbers_df, dates)
        gaps_df = pd.DataFrame({"Number": list(gaps.keys()), "Gap": list(gaps.values())}).sort_values(by="Gap", ascending=False)
        overdue_threshold = st.slider("Gap threshold for overdue numbers (draws)", min_value=0, max_value=100, value=27)
        st.dataframe(gaps_df[gaps_df["Gap"] >= overdue_threshold])

        # --- Draw Pattern Visualization: Heatmap ---
        if dates is not None:
            numbers_df_with_dates = numbers_df.copy()
            numbers_df_with_dates['Date'] = dates
            numbers_df_with_dates = numbers_df_with_dates.dropna(subset=['Date']).reset_index(drop=True)
            numbers_df_with_dates['Year'] = numbers_df_with_dates['Date'].dt.year

            years = sorted(numbers_df_with_dates['Year'].unique())
            numbers = list(range(1, 50))
            freq_matrix = pd.DataFrame(0, index=numbers, columns=years)

            for year in years:
                yearly_data = numbers_df_with_dates[numbers_df_with_dates['Year'] == year]
                nums = yearly_data.iloc[:, 0:6].values.flatten()
                counts = Counter(nums)
                for num in counts:
                    freq_matrix.at[num, year] = counts[num]

            st.subheader("Heatmap: Number Frequency by Year")
            fig_heat, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(freq_matrix, cmap="YlGnBu", linewidths=0.5, ax=ax, cbar_kws={'label': 'Frequency'})
            ax.set_xlabel("Year")
            ax.set_ylabel("Number")
            st.pyplot(fig_heat)

            # Line chart of trends for selected numbers
            st.subheader("Frequency Trend of Selected Numbers Over Years")
            selected_numbers = st.multiselect("Select numbers (1 to 49) for trend", options=range(1, 50), default=[7,14,23])
            if selected_numbers:
                trend_df = freq_matrix.loc[selected_numbers].transpose()
                fig_line = px.line(trend_df, x=trend_df.index, y=trend_df.columns,
                                   labels={"x": "Year", "value": "Frequency"},
                                   title="Number Frequency Trends Over Years")
                st.plotly_chart(fig_line, use_container_width=True)

        # --- Predictive Model ---
        st.subheader("Predictive Model: Next Draw Number Likelihood")

        with st.spinner("Training predictive model..."):
            df_feat = build_prediction_features(numbers_df)
            model, acc = train_predictive_model(df_feat)

        st.write(f"Model trained. Accuracy on test set: **{acc:.2%}**")

        probs = predict_next_draw_probs(model, numbers_df)
        probs_df = pd.DataFrame(list(probs.items()), columns=["Number", "Probability"]).sort_values(by="Probability", ascending=False)

        fig_pred = px.bar(probs_df, x="Number", y="Probability", title="Predicted Probability of Number in Next Draw", color="Probability", color_continuous_scale="Viridis")
        st.plotly_chart(fig_pred, use_container_width=True)

        # --- Ticket Generation ---
        budget = st.slider("Budget ($)", min_value=3, max_value=300, value=30, step=3)
        price_per_ticket = 3
        n_tickets = budget // price_per_ticket

        strategy = st.radio("Ticket generation strategy:", ["Hot/Cold mix", "Weighted by Frequency", "Advanced"])

        tickets = []
        if strategy == "Hot/Cold mix":
            tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
        elif strategy == "Weighted by Frequency":
            tickets = generate_tickets_weighted(counter, n_tickets)
        else:
            exclude_last_n = st.number_input("Exclude last N draws", min_value=0, max_value=30, value=2)
            recent_numbers = set(numbers_df.tail(exclude_last_n).values.flatten()) if exclude_last_n > 0 else set()
            fixed_numbers_input = st.text_input("Fixed numbers (comma-separated)", value="")
            fixed_numbers = set(int(x.strip()) for x in fixed_numbers_input.split(",") if x.strip().isdigit())
            due_numbers = set(gaps_df[gaps_df["Gap"] >= overdue_threshold]["Number"].tolist())
            if st.button("Generate Advanced Tickets"):
                tickets = generate_smart_tickets(n_tickets, fixed_numbers, recent_numbers, due_numbers)

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
        st.subheader("Lottery Probability")
        st.write("""
        **P(E) = Favorable Outcomes / Total Possible Outcomes**

        For Lotto 6/49:
        - Total combinations = C(49, 6) = 13,983,816
        - Your ticket matches 1 combination.
        - **P(E)** = 1 / 13,983,816 ≈ 0.00000715%.
        """)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

else:
    st.info("Please upload a CSV file with draw numbers.")
