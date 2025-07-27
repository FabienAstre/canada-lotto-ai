import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np
from datetime import datetime

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV Lotto 6/49",
    type=["csv"],
    help="CSV avec colonnes: DATE (optionnel), NUMBER DRAWN 1 à NUMBER DRAWN 6 et BONUS NUMBER",
)

def extract_numbers_and_bonus(df):
    required_main_cols = [
        "NUMBER DRAWN 1",
        "NUMBER DRAWN 2",
        "NUMBER DRAWN 3",
        "NUMBER DRAWN 4",
        "NUMBER DRAWN 5",
        "NUMBER DRAWN 6",
    ]
    bonus_col = "BONUS NUMBER"

    if not all(col in df.columns for col in required_main_cols):
        return None, None, None

    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    # Optional date column for gap analysis
    date_col = None
    for col_candidate in ['DATE', 'Draw Date', 'Draw_Date', 'Date']:
        if col_candidate in df.columns:
            date_col = col_candidate
            break

    dates = None
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce')
        except:
            dates = None

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

def compute_number_gaps(numbers_df, dates=None):
    # Compute how many draws since each number last appeared

    last_seen = {num: -1 for num in range(1, 50)}
    gaps = {num: None for num in range(1, 50)}

    # We assume numbers_df is sorted oldest to newest (check and sort if needed)
    # If dates given, use them for ordering; else use index
    if dates is not None:
        order = dates.argsort()
        numbers_df = numbers_df.iloc[order].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for num in range(1, 50):
            if last_seen[num] == -1:
                # Never seen before
                gaps[num] = idx  # Number of draws since start (or could mark as large)
            else:
                gaps[num] = idx - last_seen[num]
        for n in row.values:
            last_seen[n] = idx

    # After last draw, calculate gap for each number as (total_draws - last_seen_index)
    total_draws = len(numbers_df)
    for num in range(1, 50):
        if last_seen[num] != -1:
            gaps[num] = total_draws - 1 - last_seen[num]
        else:
            gaps[num] = total_draws  # never appeared, max gap

    return gaps

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Données complètes importées :")
        st.dataframe(df)

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("Le fichier CSV doit contenir les 6 colonnes principales 'NUMBER DRAWN 1' à 'NUMBER DRAWN 6' avec des nombres valides entre 1 et 49.")
        else:
            st.subheader("Derniers tirages :")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Bonus Numbers (derniers tirages) :")
                st.write(bonus_series.tail(30).to_list())

            # Frequency counts for main numbers — all draws
            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)

            # Frequency counts for bonus numbers
            bonus_counter = Counter(bonus_series) if bonus_series is not None else Counter()

            hot = [num for num, _ in counter.most_common(6)]
            cold = [num for num, _ in counter.most_common()[:-7:-1]]

            st.subheader("Numéros chauds :")
            st.write(", ".join(map(str, hot)))
            st.subheader("Numéros froids :")
            st.write(", ".join(map(str, cold)))

            if bonus_series is not None:
                st.subheader("Numéros bonus les plus fréquents :")
                bonus_hot = [num for num, _ in bonus_counter.most_common(6)]
                st.write(", ".join(map(str, bonus_hot)))

            # Frequency DataFrame for main numbers
            freq_df = pd.DataFrame({"Numéro": list(range(1, 50))})
            freq_df["Fréquence"] = freq_df["Numéro"].apply(lambda x: counter[x] if x in counter else 0)

            # Plot main numbers frequency
            fig = px.bar(
                freq_df,
                x="Numéro",
                y="Fréquence",
                title="Fréquence des numéros (tous les tirages importés)",
                labels={"Numéro": "Numéro", "Fréquence": "Nombre d'apparitions"},
                color="Fréquence",
                color_continuous_scale="Blues",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Hot vs Cold bar chart
            hot_df = freq_df[freq_df["Numéro"].isin(hot)]
            cold_df = freq_df[freq_df["Numéro"].isin(cold)]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=hot_df["Numéro"], y=hot_df["Fréquence"], name="Numéros chauds", marker_color="red"))
            fig2.add_trace(go.Bar(x=cold_df["Numéro"], y=cold_df["Fréquence"], name="Numéros froids", marker_color="blue"))
            fig2.update_layout(
                barmode="group",
                title="Comparaison Numéros chauds vs froids",
                xaxis_title="Numéro",
                yaxis_title="Fréquence",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # --- PAIR FREQUENCY ANALYSIS ---
            pair_counts = Counter()
            for _, row in numbers_df.iterrows():
                pairs = combinations(sorted(row.values), 2)
                pair_counts.update(pairs)

            top_pairs = pair_counts.most_common(10)
            pairs_df = pd.DataFrame(top_pairs, columns=["Pair", "Count"])
            pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")

            st.subheader("Top 10 des paires de numéros les plus fréquentes :")
            st.dataframe(pairs_df)

            fig_pairs = px.bar(
                pairs_df,
                y="Pair",
                x="Count",
                orientation='h',
                title="Fréquence des paires de numéros",
                labels={"Count": "Nombre d'apparitions", "Pair": "Paire de numéros"},
                color="Count",
                color_continuous_scale="Viridis",
            )
            fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_pairs, use_container_width=True)

            # --- Number Gaps & Patterns ---
            st.subheader("Analyse des écarts entre apparitions des numéros")

            gaps = compute_number_gaps(numbers_df, dates)

            gaps_df = pd.DataFrame({
                "Numéro": list(gaps.keys()),
                "Écarts (nombre de tirages depuis la dernière apparition)": list(gaps.values())
            })

            # Show the numbers with highest gaps (overdue numbers)
            overdue_threshold = st.slider(
                "Seuil d'écart minimum pour considérer un numéro comme 'en retard' (tirages)", min_value=0, max_value=100, value=20)

            overdue_df = gaps_df[gaps_df["Écarts (nombre de tirages depuis la dernière apparition)"] >= overdue_threshold]
            overdue_df = overdue_df.sort_values(by="Écarts (nombre de tirages depuis la dernière apparition)", ascending=False)

            st.write(f"Numéros en retard (écarts ≥ {overdue_threshold} tirages) :")
            st.dataframe(overdue_df)

            # Ticket generation settings
            budget = st.slider("Budget en $", min_value=3, max_value=300, value=30, step=3)
            price_per_ticket = 3
            n_tickets = budget // price_per_ticket

            # Choose ticket generation strategy
            strategy = st.radio("Choisir la méthode de génération des tickets :", 
                                ("Hot/Cold mix (original)", "Weighted by Frequency (new)"))

            def generate_tickets_hot_cold(hot, cold, n_tickets):
                tickets = set()
                pool = 49
                total_needed = 6

                while len(tickets) < n_tickets:
                    n_hot = random.randint(2, min(4, len(hot)))
                    n_cold = random.randint(2, min(4, len(cold)))

                    pick_hot = random.sample(hot, n_hot)
                    pick_cold = random.sample(cold, n_cold)

                    current = set(pick_hot + pick_cold)
                    while len(current) < total_needed:
                        current.add(random.randint(1, pool))

                    ticket_tuple = tuple(sorted(int(x) for x in current))
                    tickets.add(ticket_tuple)

                return list(tickets)

            def generate_tickets_weighted(counter, n_tickets):
                numbers = np.array(range(1, 50))
                freqs = np.array([counter.get(num, 0) for num in numbers])
                weights = freqs + 1  # Add 1 to avoid zero weights

                tickets = set()
                while len(tickets) < n_tickets:
                    ticket_np = np.random.choice(numbers, 6, replace=False, p=weights/weights.sum())
                    ticket = tuple(sorted(int(x) for x in ticket_np))  # Convert to int to avoid np.int64 print
                    tickets.add(ticket)
                return list(tickets)

            # Generate tickets based on strategy
            if strategy == "Hot/Cold mix (original)":
                tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
            else:
                tickets = generate_tickets_weighted(counter, n_tickets)

            st.subheader("Tickets générés :")
            for i, t in enumerate(tickets, 1):
                st.write(f"{i}: {t}")

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")

else:
    st.info("Veuillez importer un fichier CSV avec les numéros des tirages.")
