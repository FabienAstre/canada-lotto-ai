import streamlit as st
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt

# ------------------- CONFIG ------------------- #
st.set_page_config(page_title="Canada Lotto 6/49", page_icon="🎲")

# ------------------- FUNCTIONS ------------------- #
def load_lotto_csv(csv_file):
    """Load Lotto 6/49 data from CSV and skip date columns."""
    try:
        df = pd.read_csv(csv_file)
        df = df.dropna(how="any")  # Remove empty rows

        # Try to extract numeric columns automatically
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        if numeric_df.shape[1] < 6:
            # If numbers are strings (e.g., "01"), convert manually
            all_data = []
            for _, row in df.iterrows():
                # Take columns 1-6 (skipping the first column if it's a date)
                try:
                    numbers = [int(x) for x in row[1:7]]
                    all_data.append(numbers)
                except:
                    continue
            return all_data
        else:
            draws = numeric_df.iloc[:, :6].astype(int).values.tolist()
            return draws
    except Exception as e:
        st.error(f"Erreur CSV: {e}")
        return []


def generate_tickets(hot, cold, budget, game="649"):
    """Generate lotto tickets using hot/cold number strategy."""
    price = 3  # Price per ticket for 6/49
    n_tickets = budget // price
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

        tickets.add(tuple(sorted(current)))

    return list(tickets)


# ------------------- UI ------------------- #
st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

# ---- CSV Upload ---- #
csv_file = st.file_uploader("Importer un fichier CSV Lotto 6/49", type=["csv"])

if csv_file:
    draws = load_lotto_csv(csv_file)

    if draws:
        st.subheader("Derniers tirages :")
        df_draws = pd.DataFrame(draws, columns=[f"N{i+1}" for i in range(6)])
        st.dataframe(df_draws)

        # Frequency Analysis
        all_numbers = [num for draw in draws for num in draw]
        counter = Counter(all_numbers)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num in sorted(counter, key=counter.get)[:6]]

        st.subheader("Numéros chauds :")
        st.write(", ".join(map(str, hot)))
        st.subheader("Numéros froids :")
        st.write(", ".join(map(str, cold)))

        # Plot frequency
        st.subheader("Fréquence des numéros")
        freq_df = pd.DataFrame(counter.items(), columns=["Numéro", "Fréquence"]).sort_values(by="Fréquence", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(freq_df["Numéro"].astype(str), freq_df["Fréquence"], color="skyblue")
        ax.set_title("Fréquence des numéros (basée sur les tirages)")
        st.pyplot(fig)

        # Ticket generation
        budget = st.slider("Budget en $", 3, 100, 30, 3)
        tickets = generate_tickets(hot, cold, budget, "649")

        st.subheader("Tickets générés :")
        for i, t in enumerate(tickets, 1):
            st.write(f"{i}: {t}")
    else:
        st.warning("Le CSV est vide ou invalide.")
else:
    st.info("Veuillez importer un fichier CSV contenant les tirages du Lotto 6/49.")
