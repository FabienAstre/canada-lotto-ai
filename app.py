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
    "Importer un fichier CSV Lotto 6/49 (sans colonne DRAW DATE)",
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

    if not all(col in df.columns for col in required_main_cols):
        return None, None

    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None

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
            st.error("Le fichier CSV doit contenir au moins 6 colonnes de numéros valides entre 1 et 49.")
        else:
            st.subheader("Derniers tirages :")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Numéros Bonus (derniers tirages) :")
                st.write(bonus_series.tail(30).to_list())

            all_numbers = numbers_df.values.flatten()
            counter = Counter(all_numbers)

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

            freq_df = pd.DataFrame({"Numéro": list(range(1, 50))})
