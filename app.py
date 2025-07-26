import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV Lotto 6/49",
    type=["csv"],
    help="CSV avec colonnes: NUMBER DRAWN 1 à NUMBER DRAWN 6 et BONUS NUMBER (optionnel)",
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
    
    # Vérification des colonnes principales
    if not all(col in df.columns for col in required_main_cols):
        return None, None, "Le fichier CSV doit contenir les colonnes 'NUMBER DRAWN 1' à 'NUMBER DRAWN 6'."
    
    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce')
    valid_mask = main_numbers_df.applymap(lambda x: 1 <= x <= 49 if pd.notnull(x) else False)
    if not valid_mask.all().all():
        return None, None, "Les numéros doivent être des entiers entre 1 et 49."
    
    main_numbers_df = main_numbers_df.dropna()

    bonus_col = "BONUS NUMBER"
    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce')
        if not bonus_series.between(1, 49).all():
            bonus_series = None  # On ignore le bonus si invalide
    else:
        bonus_series = None  # Pas de colonne bonus

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Données complètes importées :")
        st.dataframe(df)

        numbers_df, bonus_series, error_msg = extract_numbers_and_bonus(df)
        if error_msg:
            st.error(error_msg)
        elif numbers_df is None:
            st.error("Erreur lors de l'extraction des numéros principaux.")
        else:
            st.subheader("Derniers tirages :")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Bonus Numbers (derniers tirages) :")
                st.write(bonus_series.tail(30).to_list())

            # Comptage des fréquences des numéros principaux
            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)

            # Comptage des fréquences des numéros bonus
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

            # DataFrame des fréquences des numéros principaux
            freq_df = pd.DataFrame({"Numéro": list(range(1, 50))})
            freq_df["Fréquence"] = freq_df["Numéro"].apply(lambda x: counter[x] if x in counter else 0)

            # Graphique fréquence numéros principaux
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
