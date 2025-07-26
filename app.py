import streamlit as st
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Canada Lotto AI", page_icon="🎲")

st.title("🇨🇦 Canada Lotto AI - Lotto 6/49")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

def fetch_lotto649_results():
    url = "https://www.getmylottoresults.com/lotto-649-past-winning-numbers/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }
    results = []
    try:
        page = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        # Essaie de sélectionner la table des résultats
        rows = soup.select("table#results tbody tr")  # Parfois c'est "results"
        if not rows:
            rows = soup.select("table.results tbody tr")  # Sinon fallback

        # Prendre max 30 tirages
        for row in rows[:30]:
            # Colonne avec les numéros gagnants
            nums_td = row.select_one("td.results")
            if nums_td:
                balls = nums_td.select("span.ball")
                nums = [int(ball.text.strip()) for ball in balls if ball.text.strip().isdigit()]
                if len(nums) >= 6:
                    results.append(nums[:6])

        return results
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return []

game = "Lotto 6/49"

with st.spinner("Récupération des derniers tirages..."):
    draws = fetch_lotto649_results()

if draws:
    st.subheader(f"Derniers tirages ({game}):")
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

    st.subheader("Statistiques avancées")
    freq_df = pd.DataFrame(counter.items(), columns=["Numéro", "Fréquence"]).sort_values(by="Fréquence", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(freq_df["Numéro"].astype(str), freq_df["Fréquence"], color="skyblue")
    ax.set_title("Fréquence des numéros (30 derniers tirages)")
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

    budget = st.slider("Budget en $", 3, 100, 30, 3)

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

    tickets = generate_tickets(hot, cold, budget, "649")

    st.subheader("Tickets générés :")
    for i, t in enumerate(tickets, 1):
        st.write(f"{i}: {t}")

else:
    st.warning("Aucun tirage disponible pour l'instant.")
