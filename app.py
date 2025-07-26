import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    # Expect CSV with 6 columns for lotto numbers, header optional
    try:
        df = pd.read_csv(file)
        # Try to find 6 numeric columns
        number_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
        if len(number_cols) < 6:
            # fallback: take first 6 columns, convert to int
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
    price = 3  # Lotto 6/49 ticket price
    n_tickets = budget // price
    tickets = set()
    pool = 49
    total_needed = 6

    while len(tickets) < n_tickets:
        n_hot = random.randint(2, min(4, len(hot)))
        n_cold = total_needed - n_hot

        # Adjust if not enough cold numbers
        if n_cold > len(cold):
            n_cold = len(cold)
            n_hot = total_needed - n_cold

        pick_hot = random.sample(hot, n_hot)
        pick_cold = random.sample(cold, n_cold)

        current = set(pick_hot + pick_cold)

        # Fill random if missing numbers
        while len(current) < total_needed:
            current.add(random.randint(1, pool))

        # Trim if too many numbers (should not happen)
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

        # Display frequencies table
        freq_df = pd.DataFrame({"Numéro": freq.index, "Fréquence": freq.values})
        st.dataframe(freq_df.style.format({"Numéro": "{:d}", "Fréquence": "{:d}"}), width=300)

        # Hot and cold numbers
        n_hot = 6
        hot_numbers = list(freq.sort_values(ascending=False).head(n_hot).index)
        cold_numbers = list(freq.sort_values().head(n_hot).index)

        st.markdown(f"**Numéros chauds :** {', '.join(map(str, hot_numbers))}")
        st.markdown(f"**Numéros froids :** {', '.join(map(str, cold_numbers))}")

        # Budget input
        budget = st.number_input("Budget en $ (chaque ticket coûte 3$)", min_value=3, max_value=300, value=30, step=3)

        tickets = generate_tickets(hot_numbers, cold_numbers, budget)

        st.subheader(f"Tickets générés ({len(tickets)}) :")
        for i, ticket in enumerate(tickets, 1):
            st.write(f"{i}: {ticket}")

        # --- Charts ---
        st.subheader("Visualisation des fréquences")

        # Frequency Bar Chart
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.bar(freq.index, freq.values, color='skyblue')
        ax1.set_xlabel("Numéro")
        ax1.set_ylabel("Fréquence")
        ax1.set_title("Fréquence des numéros tirés")
        plt.xticks(range(1, 50))
        st.pyplot(fig1)

        # Hot vs Cold Pie Chart
        fig2, ax2 = plt.subplots()
        counts = [sum(freq.loc[hot_numbers]), sum(freq.loc[cold_numbers])]
        ax2.pie(counts, labels=["Chauds", "Froids"], autopct="%1.1f%%", colors=['red', 'blue'])
        ax2.set_title("Proportion des tirages: Numéros chauds vs froids")
        st.pyplot(fig2)

        # Heatmap: Frequency by Number
        st.subheader("Carte thermique des fréquences")

        heatmap_data = np.zeros((7, 7))  # 49 numbers in 7x7 grid (7*7=49)
        for num, count in freq.items():
            row = (num - 1) // 7
            col = (num - 1) % 7
            heatmap_data[row, col] = count

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        c = ax3.imshow(heatmap_data, cmap='hot', interpolation='nearest')
        ax3.set_title("Carte thermique des fréquences (1 à 49)")
        ax3.set_xticks(range(7))
        ax3.set_yticks(range(7))
        ax3.set_xticklabels([f"{i+1}" for i in range(7)])
        ax3.set_yticklabels([f"{i*7+1}-{i*7+7}" for i in range(7)])
        fig3.colorbar(c, ax=ax3)
        st.pyplot(fig3)

else:
    st.info("Veuillez importer un fichier CSV avec les tirages Lotto 6/49.")
