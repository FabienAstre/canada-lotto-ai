import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🎲 Canada Lotto 6/49 Analyzer")

uploaded_file = st.file_uploader("Importer un fichier CSV Lotto 6/49", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur de lecture CSV: {e}")
        st.stop()

    # Columns with main drawn numbers exactly as in your file
    lotto_cols = [
        'NUMBER DRAWN 1',
        'NUMBER DRAWN 2',
        'NUMBER DRAWN 3',
        'NUMBER DRAWN 4',
        'NUMBER DRAWN 5',
        'NUMBER DRAWN 6',
    ]

    missing_cols = [col for col in lotto_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Le fichier CSV est manquant les colonnes: {', '.join(missing_cols)}")
        st.stop()

    # Extract all drawn numbers into one flat list
    all_numbers = df[lotto_cols].values.flatten()

    # Compute frequency counts and sort by number
    freq = pd.Series(all_numbers).value_counts().sort_index()

    st.subheader("Fréquence des numéros tirés (sans le numéro bonus)")
    st.write(freq.to_frame(name="Fréquence"))

    # Plot frequency bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(freq.index, freq.values, color='royalblue', label='Fréquence des tirages')
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Fréquence")
    ax.set_title("Fréquence des numéros dans les tirages importés")
    ax.legend()

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    st.pyplot(fig)
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
