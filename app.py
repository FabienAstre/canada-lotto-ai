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
    help="CSV avec colonnes des 6 numéros (N1 à N6) ou simplement 6 colonnes numériques",
)

def clean_and_extract_numbers(df):
    # Keep only columns that can be converted to int and contain values 1-49
    nums_cols = []
    for col in df.columns:
        try:
            col_vals = df[col].dropna().astype(int)
            if col_vals.between(1, 49).all():
                nums_cols.append(col)
        except Exception:
            continue
    if len(nums_cols) < 6:
        return None
    return df[nums_cols[:6]].astype(int)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        numbers_df = clean_and_extract_numbers(df)
        if numbers_df is None:
            st.error("Le fichier CSV doit contenir au moins 6 colonnes de numéros valides entre 1 et 49.")
        else:
            st.subheader("Derniers tirages :")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            # Flatten numbers for frequency
            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)

            # Hot and cold numbers
            hot = [num for num, _ in counter.most_common(6)]
            cold = [num for num, _ in counter.most_common()[:-7:-1]]  # 6 least common

            st.subheader("Numéros chauds :")
            st.write(", ".join(map(str, hot)))
            st.subheader("Numéros froids :")
            st.write(", ".join(map(str, cold)))

            # Frequency DataFrame for plotting
            freq_df = pd.DataFrame({"Numéro": list(range(1, 50))})
            freq_df["Fréquence"] = freq_df["Numéro"].apply(lambda x: counter[x] if x in counter else 0)

            # Interactive Frequency Bar Chart
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

            # Hot vs Cold Numbers Chart
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

            # Budget slider and ticket generator
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

                    # Convert all numbers to plain Python int before tuple
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
