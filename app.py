import streamlit as st
import pandas as pd
import random
import requests
from collections import Counter
import matplotlib.pyplot as plt

# ------------------- CONFIG ------------------- #
st.set_page_config(page_title="Lotto Dashboard", page_icon="🎲")

API_KEY = "YOUR_API_KEY"  # Replace with your Downtack API key
BASE_URL = "https://downtack.com/api"

# ------------------- FUNCTIONS ------------------- #

def load_canada_csv(csv_file):
    """Load Canadian Lotto 6/49 data from CSV"""
    try:
        df = pd.read_csv(csv_file)
        df = df.dropna(how="any")  # Clean empty rows
        draws = df.iloc[:, 1:7].values.tolist()  # Columns with numbers
        draws = [[int(num) for num in row] for row in draws]
        return draws
    except Exception as e:
        st.error(f"Erreur CSV: {e}")
        return []

def get_us_lottery_games():
    """Fetch U.S. lottery games from Downtack API"""
    url = f"{BASE_URL}/get-all-games?api_key={API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

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

# ------------------- UI ------------------- #

st.title("🎲 Lotto Dashboard")
st.write("Analyse des tirages pour le Canada (Lotto 6/49) et USA (Powerball, Mega Millions).")

# ---- Canada Lotto ---- #
st.header("🇨🇦 Canada Lotto 6/49")
canada_csv = st.file_uploader("Importer un CSV Lotto 6/49", type=["csv"])

if canada_csv:
    draws = load_canada_csv(canada_csv)

    if draws:
        st.subheader("30 derniers tirages :")
        df_draws = pd.DataFrame(draws, columns=[f"N{i+1}" for i in range(len(draws[0]))])
        st.dataframe(df_draws)

        all_numbers = [num for draw in draws for num in draw]
        counter = Counter(all_numbers)
        hot = [num for num, _ in counter.most_common(6)]
        cold = [num for num in sorted(counter, key=counter.get)[:6]]

        st.subheader("Numéros chauds :")
        st.write(", ".join(map(str, hot)))
        st.subheader("Numéros froids :")
        st.write(", ".join(map(str, cold)))

        st.subheader("Statistiques (30 derniers tirages)")
        freq_df = pd.DataFrame(counter.items(), columns=["Numéro", "Fréquence"]).sort_values(by="Fréquence", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(freq_df["Numéro"].astype(str), freq_df["Fréquence"], color="skyblue")
        ax.set_title("Fréquence des numéros")
        st.pyplot(fig)

        budget = st.slider("Budget en $", 3, 100, 30, 3)
        tickets = generate_tickets(hot, cold, budget, "649")

        st.subheader("Tickets générés :")
        for i, t in enumerate(tickets, 1):
            st.write(f"{i}: {t}")
    else:
        st.warning("Le CSV est vide ou invalide.")
else:
    st.info("Veuillez importer un fichier CSV contenant les tirages du Lotto 6/49.")

# ---- U.S. Lotteries ---- #
st.header("🇺🇸 U.S. Lotteries (via Downtack API)")
if st.button("Charger les résultats U.S."):
    try:
        us_data = get_us_lottery_games()
        st.success("Données U.S. récupérées avec succès !")

        # Display only Powerball & Mega Millions
        for state, games in us_data.items():
            for game in games:
                if game['name'].lower() in ["powerball", "mega millions"]:
                    st.subheader(f"{game['name']} ({state})")
                    latest_draw = game['plays'][0]['draws'][0]
                    numbers = [n['value'] for n in latest_draw['numbers']]
                    st.write(f"Date : {latest_draw['date']}")
                    st.write(f"Numéros : {', '.join(numbers)}")
    except Exception as e:
        st.error(f"Erreur Downtack API: {e}")
