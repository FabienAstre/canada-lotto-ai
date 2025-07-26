import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.markdown("""
Analyse des tirages réels, statistiques et génération de tickets.
""")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV Lotto 6/49\n(Limite 200MB)",
    type=["csv"]
)

def load_data(file):
    try:
        df = pd.read_csv(file)
        # Assume first column might be date
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        else:
            # Try parse first col as date
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            except Exception:
                pass

        # Find columns with lottery numbers (ints)
        number_cols = []
        for col in df.columns:
            if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                number_cols.append(col)

        # We expect 6 numbers per draw
        if len(number_cols) < 6:
            # fallback to first 6 columns after date
            df = df.iloc[:, 1:7]
        else:
            df = df[number_cols[:6]]

        df.columns = ["N1", "N2", "N3", "N4", "N5", "N6"]
        df.dropna(inplace=True)
        df = df.astype(int)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSV: {e}")
        return None

def number_frequencies(df):
    all_numbers = pd.Series(df.values.flatten())
    freq = all_numbers.value_counts().sort_index()
    return freq

def generate_tickets(hot, cold, budget):
    price = 3
    n_tickets = budget // price
    tickets = set()
    pool = 49
    total_needed = 6

    while len(tickets) < n_tickets:
        n_hot = random.randint(2, min(4, len(hot)))
        n_cold = total_needed - n_hot

        if n_cold > len(cold):
            n_cold = len(cold)
            n_hot = total_needed - n_cold

        pick_hot = random.sample(hot, n_hot)
        pick_cold = random.sample(cold, n_cold)

        current = set(pick_hot + pick_cold)

        while len(current) < total_needed:
            current.add(random.randint(1, pool))

        if len(current) > total_needed:
            current = set(random.sample(current, total_needed))

        tickets.add(tuple(sorted(current)))

    return list(tickets)

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None and not df.empty:
        st.subheader("Derniers tirages :")
        st.dataframe(df.tail(12), width=700)

        freq = number_frequencies(df)
        st.subheader("Fréquence des numéros tirés")

        freq_df = pd.DataFrame({"Numéro": freq.index, "Fréquence": freq.values})
        st.dataframe(freq_df.style.format({"Numéro": "{:d}", "Fréquence": "{:d}"}), width=300)

        n_hot = 6
        hot_numbers = list(freq.sort_values(ascending=False).head(n_hot).index)
        cold_numbers = list(freq.sort_values().head(n_hot).index)

        st.markdown(f"**Numéros chauds :** {', '.join(map(str, hot_numbers))}")
        st.markdown(f"**Numéros froids :** {', '.join(map(str, cold_numbers))}")

        budget = st.number_input(_
