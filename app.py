import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyse des tirages, statistiques et génération de tickets.")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV Lotto 6/49",
    type=["csv"],
    help="CSV avec colonnes : NUMBER DRAWN 1 à NUMBER DRAWN 6 et BONUS NUMBER (la colonne DRAW DATE est optionnelle et ignorée)",
)

def extract_numbers_and_bonus(df):
    # Nettoyer noms colonnes
    df.columns = df.columns.str.strip().str.upper()
    
    required_main_cols = [
        "NUMBER DRAWN 1",
        "NUMBER DRAWN 2",
        "NUMBER DRAWN 3",
        "NUMBER DRAWN 4",
        "NUMBER DRAWN 5",
        "NUMBER DRAWN 6",
    ]
    bonus_col = "BONUS NUMBER"
    
    missing_cols = [col for col in required_main_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Colonnes principales manquantes : {missing_cols}")
        return None, None

    # Convertir colonnes principales en numérique et vérifier NaN
    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce')
    if main_numbers_df.isna().any().any():
        st.error("Des valeurs invalides ou manquantes détectées dans les colonnes principales.")
        st.write(main_numbers_df[main_numbers_df.isna().any(axis=1)])
        return None, None
    
    # Vérifier valeurs entre 1 et 49
    if not ((main_numbers_df >= 1) & (main_numbers_df <= 49)).all().all():
        st.error("Toutes les valeurs des colonnes principales doivent être entre 1 et 49.")
        invalid = main_numbers_df[~((main_numbers_df >= 1) & (main_numbers_df <= 49)).all(axis=1)]
        st.write(invalid)
        return None, None

    # Bonus
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce')
        if bonus_series.isna().any():
            st.error("Des valeurs invalides détectées dans la colonne BONUS NUMBER.")
            return None, None
        if not bonus_series.between(1, 49).all():
            st.error("Toutes les valeurs de BONUS NUMBER doivent être entre 1 et 49.")
            return None, None

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Données complètes importées :")
        st.dataframe(df.head(10))
        
        numbers_df, bonus_series = extract_numbers_and_bonus(df)

        if numbers_df is not None:
            st.subheader("Derniers tirages (numéros) :")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Bonus Numbers (derniers tirages) :")
                st.write(bonus_series.tail(30).to_list())

            # Toutes les valeurs des tirages principaux à plat
            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)

            # Bonus counter
            bonus_counter = Counter(bonus_series) if bonus_series is not None else Counter()

            hot = [num for num, _ in counter.most_common(6)]
            cold = [num for num, _ in counter.most_common()[:-7:-1]]

            st.subheader("Numéros chauds (plus fréquents) :")
            st.write(", ".join(map(str, hot)))
            st.subheader("Numéros froids (moins fréquents) :")
            st.write(", ".join(map(str, cold)))

            if bonus_series is not None:
                st.subheader("Numéros bonus les plus fréquents :")
                bonus_hot = [num for num, _ in bonus_counter.most_common(6)]
                st.write(", ".join(map(str, bonus_hot)))

            # DataFrame fréquence tous numéros 1-49
            freq_df = pd.DataFrame({"Numéro": list(range(1, 50))})
            freq_df["Fréquence"] = freq_df["Numéro"].apply(lambda x: counter[x] if x in counter else 0)

            # Graphe fréquence
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

            # Graphique chaud vs froid
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

            # Analyse des paires
            pair_counts = Counter()
            for _, row in numbers_df.iterrows():
                pairs = combinations(sorted(row.values), 2)
                pair_counts.update(pairs)

            top_pairs = pair_counts.most_common(10)
            pairs_df = pd.DataFrame(top_pairs, columns=["Paire", "Nombre d'apparitions"])
            pairs_df["Paire"] = pairs_df["Paire"].apply(lambda x: f"{x[0]} & {x[1]}")

            st.subheader("Top 10 des paires de numéros les plus fréquentes :")
            st.dataframe(pairs_df)

            fig_pairs = px.bar(
                pairs_df,
                y="Paire",
                x="Nombre d'apparitions",
                orientation='h',
                title="Fréquence des paires de numéros",
                labels={"Nombre d'apparitions": "Nombre d'apparitions", "Paire": "Paire de numéros"},
                color="Nombre d'apparitions",
                color_continuous_scale="Viridis",
            )
            fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_pairs, use_container_width=True)

            # Génération tickets
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

