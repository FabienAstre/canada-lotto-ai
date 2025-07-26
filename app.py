
import streamlit as st
import pandas as pd
import random
from collections import Counter

st.set_page_config(page_title="Canada Lotto AI", page_icon="🎲")

st.title("🇨🇦 Canada Lotto AI - 6/49 & Lotto Max")
st.write("Analyse des numéros chauds et froids, et génération de tickets.")

# Choix du jeu
game = st.selectbox("Choisir le jeu :", ["Lotto 6/49", "Lotto Max"])
budget = st.slider("Budget en $", 3, 100, 30, 3)

# Exemple de numéros chauds/froids
def get_hot_cold(game="649"):
    # Ici on simule quelques numéros fréquents et rares (à remplacer par analyse réelle)
    hot = [6, 11, 15, 27, 32, 43]
    cold = [2, 7, 36, 38, 49, 13]
    return hot, cold

hot, cold = get_hot_cold("649" if "6/49" in game else "max")

st.subheader("Numéros chauds :")
st.write(", ".join(map(str, hot)))

st.subheader("Numéros froids :")
st.write(", ".join(map(str, cold)))

# Génération des tickets
def generate_tickets(hot, cold, budget, game="649"):
    price = 3 if game == "649" else 5
    n_tickets = budget // price
    tickets = []
    for _ in range(n_tickets):
        pick_hot = random.sample(hot, min(3, len(hot)))
        pick_cold = random.sample(cold, min(3, len(cold)))
        total_needed = 6 if game == "649" else 7
        while len(pick_hot) + len(pick_cold) < total_needed:
            pool = 49 if game == "649" else 50
            new_num = random.randint(1, pool)
            if new_num not in pick_hot and new_num not in pick_cold:
                pick_cold.append(new_num)
        tickets.append(sorted(pick_hot + pick_cold))
    return tickets

game_type = "649" if "6/49" in game else "max"
tickets = generate_tickets(hot, cold, budget, game_type)

st.subheader("Tickets générés :")
for i, t in enumerate(tickets, 1):
    st.write(f"{i}: {t}")
