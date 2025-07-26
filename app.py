
import streamlit as st
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Canada Lotto AI", page_icon="🎲")

st.title("🇨🇦 Canada Lotto AI - 6/49 & Lotto Max")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

# --- Scraping des résultats ---
def fetch_lotto_results(game="649"):
    if game == "649":
        url = "https://www.lotterycanada.com/649/results"
    else:
        url = "https://www.lotterycanada.com/lotto-max/results"
    try:
        page = requests.get(url, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")
        results = []
        for draw in soup.select(".winning-numbers"):
            nums = [n.text for n in draw.select("li")]
            nums = [int(n) for n in nums if n.isdigit()]
            if len(nums) >= 6:
                results.append(nums[:6] if game == "649" else nums[:7])
        return results[:30]  # On prend les 30 derniers tirages
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return []

game = st.selectbox("Choisir le jeu :", ["Lotto 6/49", "Lotto Max"])
game_type = "649" if "6/49" in game else "max"

with st.spinner("Récupération des derniers tirages..."):
    draws = fetch_lotto_results(game_type)

if draws:
    st.subheader(f"Derniers tirages ({game}):")
    df_draws = pd.DataFrame(draws, columns=[f"N{i+1}" for i in range(len(draws[0]))])
    st.dataframe(df_draws)

    # Analyse des numéros chauds/froids
    all_numbers = [num for draw in draws for num in draw]
    counter = Counter(all_numbers)
    hot = [num for num, _ in counter.most_common(6)]
    cold = [num for num, _ in counter.most_common()[-6:]]

    st.subheader("Numéros chauds :")
    st.write(", ".join(map(str, hot)))
    st.subheader("Numéros froids :")
    st.write(", ".join(map(str, cold)))

    # --- Statistiques avancées ---
    st.subheader("Statistiques avancées")
    freq_df = pd.DataFrame(counter.items(), columns=["Numéro", "Fréquence"]).sort_values(by="Fréquence", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(freq_df["Numéro"].astype(str), freq_df["Fréquence"], color="skyblue")
    ax.set_title("Fréquence des numéros (30 derniers tirages)")
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

    # --- Génération des tickets ---
    budget = st.slider("Budget en $", 3, 100, 30, 3)

    def generate_tickets(hot, cold, budget, game="649"):
        price = 3 if game == "649" else 5
        n_tickets = budget // price
        tickets = []
        pool = 49 if game == "649" else 50
        total_needed = 6 if game == "649" else 7
        for _ in range(n_tickets):
            pick_hot = random.sample(hot, min(3, len(hot)))
            pick_cold = random.sample(cold, min(3, len(cold)))
            current = set(pick_hot + pick_cold)
            while len(current) < total_needed:
                current.add(random.randint(1, pool))
            tickets.append(sorted(current))
        return tickets

    tickets = generate_tickets(hot, cold, budget, game_type)
    st.subheader("Tickets générés :")
    for i, t in enumerate(tickets, 1):
        st.write(f"{i}: {t}")
else:
    st.warning("Aucun tirage disponible pour l'instant.")
