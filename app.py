import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors

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
        number_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
        if len(number_cols) < 6:
            df = df.iloc[:, :6]
            df = df.apply(pd.to_numeric, errors='coerce')
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

        budget = st.number_input("Budget en $ (chaque ticket coûte 3$)", min_value=3, max_value=300, value=30, step=3)

        tickets = generate_tickets(hot_numbers, cold_numbers, budget)

        st.subheader(f"Tickets générés ({len(tickets)}) :")
        for i, ticket in enumerate(tickets, 1):
            st.write(f"{i}: {ticket}")

        st.subheader("Visualisation des fréquences")

        # Sorted bar chart with color gradient
        sorted_freq = freq.sort_values(ascending=True)
        colors = plt.cm.viridis((sorted_freq - sorted_freq.min()) / (sorted_freq.max() - sorted_freq.min()))

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.barh(sorted_freq.index, sorted_freq.values, color=colors)
        ax1.set_xlabel("Fréquence")
        ax1.set_ylabel("Numéro")
        ax1.set_title("Fréquence des numéros tirés (trié par fréquence)")
        st.pyplot(fig1)

        # Cumulative frequency line chart
        cum_freq = freq.sort_index().cumsum()
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(cum_freq.index, cum_freq.values, marker='o', linestyle='-', color='orange')
        ax2.set_xlabel("Numéro")
        ax2.set_ylabel("Fréquence cumulée")
        ax2.set_title("Fréquence cumulée des numéros tirés (1 à 49)")
        ax2.grid(True)
        st.pyplot(fig2)

        # Hot vs Cold Pie Chart
        fig3, ax3 = plt.subplots()
        counts = [sum(freq.loc[hot_numbers]), sum(freq.loc[cold_numbers])]
        ax3.pie(counts, labels=["Chauds", "Froids"], autopct="%1.1f%%", colors=['red', 'blue'])
        ax3.set_title("Proportion des tirages: Numéros chauds vs froids")
        st.pyplot(fig3)

else:
    st.info("Veuillez importer un fichier CSV avec les tirages Lotto 6/49.")
