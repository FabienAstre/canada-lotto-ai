import streamlit as st
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Canada Lotto AI", page_icon="🎲")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV Lotto 6/49",
    type=["csv"],
    accept_multiple_files=False,
)

def process_draws(df):
    # Ensure columns are int and drop invalid rows
    try:
        draws = df.astype(int)
        # Keep only first 6 columns if more are present
        draws = draws.iloc[:, :6]
        return draws
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {e}")
        return None

def generate_tickets(hot, cold, budget, game="649"):
    price = 3 if game == "649" else 5
    n_tickets = budget // price
    tickets = set()
    pool = 49 if game == "649" else 50
    total_needed = 6 if game == "649" else 7

    while len(tickets) < n_tickets:
        n_hot = random.randint(2, min(4, len(hot)))
        n_cold = random.randint(2, min(4, len(cold)))

        pick_hot = random.sample(hot, n_hot)
        pick_cold = random.sample(cold, n_cold)

        current = set(pick_hot + pick_cold)
        while len(current) < total_needed:
            current.add(random.randint(1, pool))

        tickets.add(tuple(sorted(current)))

    return list(tickets)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        draws = process_draws(df)
        if draws is None or draws.empty:
            st.warning("Le fichier CSV est vide ou invalide.")
        else:
            st.subheader("Derniers tirages :")
            st.dataframe(draws)

            # Flatten all numbers for frequency analysis
            all_numbers = draws.values.flatten()
            counter = Counter(all_numbers)
            hot = [num for num, _ in counter.most_common(6)]
            cold = [num for num in sorted(counter, key=counter.get)[:6]]

            st.subheader("Numéros chauds :")
            st.write(", ".join(map(str, hot)))
            st.subheader("Numéros froids :")
            st.write(", ".join(map(str, cold)))

            st.subheader("Fréquence des numéros")
            freq_df = pd.DataFrame(counter.items(), columns=["Numéro", "Fréquence"]).sort_values(by="Fréquence", ascending=False)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(freq_df["Numéro"].astype(str), freq_df["Fréquence"], color="skyblue")
            ax.set_title("Fréquence des numéros (tirages importés)")
            ax.set_xlabel("Numéro")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)

            budget = st.slider("Budget en $", 3, 100, 30, 3)

            tickets = generate_tickets(hot, cold, budget, "649")

            st.subheader("Tickets générés :")
            for i, ticket in enumerate(tickets, 1):
                # Convert np.int64 to native int for clean display
                clean_ticket = tuple(int(n) for n in ticket)
                st.write(f"{i}: {clean_ticket}")

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
else:
    st.info("Veuillez importer un fichier CSV valide contenant les tirages Lotto 6/49.")
