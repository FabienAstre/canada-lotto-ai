import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV Lotto 6/49",
    type=["csv"],
    help="CSV avec colonnes: NUMBER DRAWN 1 à NUMBER DRAWN 6 et BONUS NUMBER",
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

    # Check required columns exist
    if not all(col in df.columns for col in required_main_cols):
        return None, None

    # Extract main numbers and ensure valid range
    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None

    # Extract bonus number if available and valid
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Données complètes importées :")
        st.dataframe(df)

        numbers_df, bonus_series = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("Le fichier CSV doit contenir les 6 colonnes principales 'NUMBER DRAWN 1' à 'NUMBER DRAWN 6' avec des nombres valides entre 1 et 49.")
        else:
            st.subheader("Derniers tirages (6 numéros) :")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Bonus Numbers (derniers tirages) :")
                st.write(bonus_series.tail(30).to_list())

            # Frequency counts for main numbers
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
                title="Fréquence des numéros (30 derniers tirages)",
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

            # Ticket generation
            budget = st.slider("Budget en $", min_value=3, max_value=300, value=30, step=3)
            price_per_ticket = 3
            n_tickets = budget // price_per_ticket

            def generate_tickets(hot, cold, n_tickets):
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

            tickets = generate_tickets(hot, cold, n_tickets)

            st.subheader("Tickets générés :")
            for i, t in enumerate(tickets, 1):
                st.write(f"{i}: {t}")

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")

else:
    st.info("Veuillez importer un fichier CSV avec les numéros des tirages.")
