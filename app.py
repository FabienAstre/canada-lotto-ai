import streamlit as st
import pandas as pd
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import numpy as np

st.set_page_config(page_title="🎲 Canada Lotto 6/49 Analyzer", page_icon="🎲", layout="wide")

st.title("🎲 Canada Lotto 6/49 Analyzer")
st.write("Analyse des tirages réels, statistiques et génération de tickets.")

# --- Helpers ---
def to_py_int_set(xs):
    return set(int(x) for x in xs)

def to_py_ticket(ticket):
    return tuple(sorted(int(x) for x in ticket))

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Importer un fichier CSV Lotto 6/49",
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
        return None, None, None

    main_numbers_df = df[required_main_cols].apply(pd.to_numeric, errors='coerce').dropna()
    if not main_numbers_df.applymap(lambda x: 1 <= x <= 49).all().all():
        return None, None, None

    bonus_series = None
    if bonus_col in df.columns:
        bonus_series = pd.to_numeric(df[bonus_col], errors='coerce').dropna()
        if not bonus_series.between(1, 49).all():
            bonus_series = None

    date_col = None
    for col_candidate in ['DATE', 'Draw Date', 'Draw_Date', 'Date']:
        if col_candidate in df.columns:
            date_col = col_candidate
            break

    dates = None
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce')
        except:
            dates = None

    return main_numbers_df.astype(int), bonus_series.astype(int) if bonus_series is not None else None, dates

def compute_number_gaps(numbers_df, dates=None):
    last_seen = {num: -1 for num in range(1, 50)}
    gaps = {num: None for num in range(1, 50)}

    if dates is not None:
        order = dates.argsort()
        numbers_df = numbers_df.iloc[order].reset_index(drop=True)
    else:
        numbers_df = numbers_df.reset_index(drop=True)

    for idx, row in numbers_df.iterrows():
        for num in range(1, 50):
            if last_seen[num] == -1:
                gaps[num] = idx
            else:
                gaps[num] = idx - last_seen[num]
        for n in row.values:
            last_seen[n] = idx

    total_draws = len(numbers_df)
    for num in range(1, 50):
        if last_seen[num] != -1:
            gaps[num] = total_draws - 1 - last_seen[num]
        else:
            gaps[num] = total_draws

    return gaps

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Données complètes importées :")
        st.dataframe(df)

        numbers_df, bonus_series, dates = extract_numbers_and_bonus(df)

        if numbers_df is None:
            st.error("Le fichier CSV doit contenir les 6 colonnes principales 'NUMBER DRAWN 1' à 'NUMBER DRAWN 6' avec des nombres valides entre 1 et 49.")
        else:
            st.subheader("Derniers tirages :")
            st.dataframe(numbers_df.tail(30).reset_index(drop=True))

            if bonus_series is not None:
                st.subheader("Bonus Numbers (derniers tirages) :")
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
            freq_df["Fréquence"] = freq_df["Numéro"].apply(lambda x: counter[x] if x in counter else 0)

            fig = px.bar(
                freq_df,
                x="Numéro",
                y="Fréquence",
                title="Fréquence des numéros (tous les tirages importés)",
                labels={"Numéro": "Numéro", "Fréquence": "Nombre d'apparitions"},
                color="Fréquence",
                color_continuous_scale="Blues",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            hot_df = freq_df[freq_df["Numéro"].isin(hot)]
            cold_df = freq_df[freq_df["Numéro"].isin(cold)]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=hot_df["Numéro"], y=hot_df["Fréquence"], name="Numéros chauds", marker_color="red"))
            fig2.add_trace(go.Bar(x=cold_df["Numéro"], y=cold_df["Fréquence"], name="Numéros froids", marker_color="blue"))
            fig2.update_layout(
                barmode="group",
                title="Comparaison Numéros chauds vs froids",
                xaxis_title="Numéro",
                yaxis_title="Fréquence",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

            pair_counts = Counter()
            for _, row in numbers_df.iterrows():
                pairs = combinations(sorted(row.values), 2)
                pair_counts.update(pairs)

            top_pairs = pair_counts.most_common(10)
            pairs_df = pd.DataFrame(top_pairs, columns=["Pair", "Count"])
            pairs_df["Pair"] = pairs_df["Pair"].apply(lambda x: f"{x[0]} & {x[1]}")

            st.subheader("Top 10 des paires de numéros les plus fréquentes :")
            st.dataframe(pairs_df)

            fig_pairs = px.bar(
                pairs_df,
                y="Pair",
                x="Count",
                orientation='h',
                title="Fréquence des paires de numéros",
                labels={"Count": "Nombre d'apparitions", "Pair": "Paire de numéros"},
                color="Count",
                color_continuous_scale="Viridis",
            )
            fig_pairs.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_white")
            st.plotly_chart(fig_pairs, use_container_width=True)

            st.subheader("Analyse des écarts entre apparitions des numéros")
            gaps = compute_number_gaps(numbers_df, dates)

            gaps_df = pd.DataFrame({
                "Numéro": list(gaps.keys()),
                "Écarts (nombre de tirages depuis la dernière apparition)": list(gaps.values())
            })

            overdue_threshold = st.slider(
                "Seuil d'écart minimum pour considérer un numéro comme 'en retard' (tirages)", min_value=0, max_value=100, value=20)

            overdue_df = gaps_df[gaps_df["Écarts (nombre de tirages depuis la dernière apparition)"] >= overdue_threshold]
            overdue_df = overdue_df.sort_values(by="Écarts (nombre de tirages depuis la dernière apparition)", ascending=False)

            st.write(f"Numéros en retard (écarts ≥ {overdue_threshold} tirages) :")
            st.dataframe(overdue_df)

            budget = st.slider("Budget en $", min_value=3, max_value=300, value=30, step=3)
            price_per_ticket = 3
            n_tickets = budget // price_per_ticket

            strategy = st.radio("Choisir la méthode de génération des tickets :",
                                ("Hot/Cold mix (original)", "Weighted by Frequency (new)", "Avancée (fixés, exclusions, retard)"))

            # --- Ticket Generators ---
            def generate_tickets_hot_cold(hot, cold, n_tickets):
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
                    ticket_tuple = to_py_ticket(current)
                    tickets.add(ticket_tuple)
                return list(tickets)

            def generate_tickets_weighted(counter, n_tickets):
                numbers = np.array(range(1, 50))
                freqs = np.array([counter.get(num, 0) for num in numbers])
                weights = freqs + 1
                tickets = set()
                while len(tickets) < n_tickets:
                    ticket_np = np.random.choice(numbers, 6, replace=False, p=weights/weights.sum())
                    ticket = to_py_ticket(ticket_np)
                    tickets.add(ticket)
                return list(tickets)

            def generate_smart_tickets(n_tickets, fixed_nums, exclude_nums, due_nums, top_pairs, total_nums=6):
                tickets = set()
                pool = set(range(1, 50))
                pool -= exclude_nums
                pool -= fixed_nums
                top_pairs_set = set(tuple(sorted([int(p1), int(p2)])) for (pair, _) in top_pairs for p1, p2 in [pair])
                while len(tickets) < n_tickets:
                    ticket = set(int(x) for x in fixed_nums)
                    due_to_add = set(int(x) for x in (due_nums - ticket))
                    while len(ticket) < total_nums and due_to_add:
                        ticket.add(due_to_add.pop())
                    pairs_to_consider = [p for p in top_pairs_set if any(num in ticket for num in p)]
                    random.shuffle(pairs_to_consider)
                    for p in pairs_to_consider:
                        if len(ticket) >= total_nums:
                            break
                        for num in p:
                            if num not in ticket and num not in exclude_nums and len(ticket) < total_nums:
                                ticket.add(int(num))
                    remaining_pool = list(pool - ticket)
                    random.shuffle(remaining_pool)
                    for num in remaining_pool:
                        if len(ticket) >= total_nums:
                            break
                        ticket.add(int(num))
                    ticket_tuple = to_py_ticket(ticket)
                    tickets.add(ticket_tuple)
                return list(tickets)

            # --- Advanced Strategy ---
            if strategy == "Avancée (fixés, exclusions, retard)":
                st.subheader("Options avancées de génération de tickets")

                exclude_last_n = st.number_input("Exclure les numéros tirés dans les derniers N tirages", min_value=0, max_value=30, value=2, step=1)
                recent_numbers = set()
                if exclude_last_n > 0 and len(numbers_df) >= exclude_last_n:
                    recent_draws = numbers_df.tail(exclude_last_n)
                    recent_numbers = to_py_int_set(recent_draws.values.flatten().tolist())

                include_due = st.checkbox("Inclure les numéros en retard (selon seuil d'écart défini)", value=True)
                due_numbers = set()
                if include_due:
                    due_numbers = to_py_int_set(overdue_df["Numéro"].tolist()) if not overdue_df.empty else set()

                fixed_numbers_input = st.text_input("Entrez vos numéros fixes séparés par des virgules (ex: 5,12,23)", value="")
                fixed_numbers = set()
                if fixed_numbers_input.strip():
                    try:
                        fixed_numbers = set(int(x.strip()) for x in fixed_numbers_input.split(",") if 1 <= int(x.strip()) <= 49)
                    except:
                        st.error("Veuillez entrer des numéros valides entre 1 et 49, séparés par des virgules.")
                if len(fixed_numbers) > 5:
                    st.error("Vous pouvez fixer au maximum 5 numéros. Le reste sera généré aléatoirement.")
                    fixed_numbers = set(list(fixed_numbers)[:5])

                if st.button("Générer des tickets avancés"):
                    tickets = generate_smart_tickets(
                        n_tickets=n_tickets,
                        fixed_nums=fixed_numbers,
                        exclude_nums=recent_numbers,
                        due_nums=due_numbers,
                        top_pairs=top_pairs
                    )
                    st.subheader("Tickets avancés générés :")
                    for i, t in enumerate(tickets, 1):
                        st.write(f"{i}: {t}")

            else:
                if strategy == "Hot/Cold mix (original)":
                    tickets = generate_tickets_hot_cold(hot, cold, n_tickets)
                else:
                    tickets = generate_tickets_weighted(counter, n_tickets)

                st.subheader("Tickets générés :")
                for i, t in enumerate(tickets, 1):
                    st.write(f"{i}: {t}")

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")

else:
    st.info("Veuillez importer un fichier CSV avec les numéros des tirages.")
