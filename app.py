import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🎲 Canada Lotto 6/49 Analyzer")

uploaded_file = st.file_uploader("Importer un fichier CSV Lotto 6/49", type=["csv"])

if uploaded_file is not None:
    # Reset file pointer to the start before reading
    uploaded_file.seek(0)
    try:
        # Read CSV only once
        original_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur de lecture CSV: {e}")
        st.stop()

    st.subheader("Aperçu des données originales")
    st.write(original_df.head())

    # Assuming columns are the lotto numbers columns, e.g., N1, N2, ..., N6
    lotto_columns = [col for col in original_df.columns if col.startswith('N')]
    if not lotto_columns:
        st.error("Le fichier CSV ne contient pas les colonnes attendues (N1, N2, ..., N6).")
        st.stop()

    # Extract numbers from lotto columns, flatten to a single list
    numbers = original_df[lotto_columns].values.flatten()

    # Compute frequency of each number
    freq = pd.Series(numbers).value_counts().sort_index()

    st.subheader("Fréquence des numéros")

    # Plot frequency bar chart with legend
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(freq.index, freq.values, color='skyblue', label='Fréquence')
    ax.set_xlabel("Numéros")
    ax.set_ylabel("Nombre d'apparitions")
    ax.set_title("Fréquence des numéros dans les tirages importés")
    ax.legend()

    st.pyplot(fig)

    # Display the best number by date
    if 'Date' in original_df.columns:
        st.subheader("Meilleur numéro par date")
        # Example: most frequent number per date
        best_numbers = {}
        for date, group in original_df.groupby('Date'):
            nums = group[lotto_columns].values.flatten()
            counts = pd.Series(nums).value_counts()
            best_num = counts.idxmax()
            best_numbers[date] = best_num

        best_num_df = pd.DataFrame(list(best_numbers.items()), columns=['Date', 'Meilleur numéro'])
        st.write(best_num_df)
    else:
        st.info("Colonne 'Date' non trouvée dans le fichier CSV, impossible d'afficher le meilleur numéro par date.")
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
