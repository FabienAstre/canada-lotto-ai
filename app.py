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

        budget = st.number_input("Budget en $ (chaque ticket coûte 3$)", min_value=3, max_value=300, value=30, step=3)

        tickets = generate_tickets(hot_numbers, cold_numbers, budget)

        st.subheader(f"Tickets générés ({len(tickets)}) :")
        for i, ticket in enumerate(tickets, 1):
            st.write(f"{i}: {ticket}")

        # === Frequency bar chart with color legend ===
        sorted_freq = freq.sort_values(ascending=True)
        normalized = (sorted_freq - sorted_freq.min()) / (sorted_freq.max() - sorted_freq.min())
        colors = plt.cm.viridis(normalized)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(sorted_freq.index, sorted_freq.values, color=colors)
        ax.set_xlabel("Fréquence")
        ax.set_ylabel("Numéro")
        ax.set_title("Fréquence des numéros tirés (trié par fréquence)")

        # Colorbar legend
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=sorted_freq.min(), vmax=sorted_freq.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Nombre d\'apparitions')

        st.pyplot(fig)

        # === Best numbers over time (by date) ===
        # We try to detect date from first column of uploaded CSV
        original_df = pd.read_csv(uploaded_file)
        date_col = None
        for col in original_df.columns:
            if pd.api.types.is_datetime64_any_dtype(original_df[col]) or \
               pd.to_datetime(original_df[col], errors='coerce').notnull().all():
                date_col = col
                break

        if date_col:
            # Parse date column safely
            dates = pd.to_datetime(original_df[date_col], errors='coerce')
            original_df[date_col] = dates
            # Filter rows with valid dates
            valid_rows = ~dates.isna()

            # Create a DataFrame with date and all numbers columns
            numbers_cols = [c for c in original_df.columns if c != date_col]
            draws = original_df.loc[valid_rows, [date_col] + numbers_cols].copy()

            # Melt numbers into one column per row for plotting frequency over time
            draws_long = draws.melt(id_vars=[date_col], value_vars=numbers_cols, value_name='Number')
            draws_long.dropna(subset=['Number'], inplace=True)
            draws_long['Number'] = draws_long['Number'].astype(int)

            # Count occurrences of each number per date
            freq_over_time = draws_long.groupby([date_col, 'Number']).size().reset_index(name='Count')

            # Focus on top 5 hot numbers frequency over time
            top_numbers = hot_numbers[:5]
            freq_over_time_top = freq_over_time[freq_over_time['Number'].isin(top_numbers)]

            fig2, ax2 = plt.subplots(figsize=(12, 6))
            for num in top_numbers:
                data_num = freq_over_time_top[freq_over_time_top['Number'] == num]
                ax2.plot(data_num[date_col], data_num['Count'].cumsum(), label=f"Num {num}")

            ax2.set_title("Fréquence cumulée des numéros chauds au fil du temps")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Fréquence cumulée")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
        else:
            st.info("Aucune colonne de date détectée dans le fichier CSV, le graphique 'numéros chauds par date' ne peut pas être affiché.")

else:
    st.info("Veuillez importer un fichier CSV avec les tirages Lotto 6/49.")
