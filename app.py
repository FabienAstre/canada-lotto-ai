import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🎲 Canada Lotto 6/49 Analyzer")

uploaded_file = st.file_uploader("Importer un fichier CSV Lotto 6/49", type=["csv"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erreur de lecture CSV: {e}")
        st.stop()

    # Detect columns with lotto numbers (N1, N2, ..., N6)
    lotto_cols = [col for col in df.columns if col.strip().upper() in ['N1','N2','N3','N4','N5','N6']]
    if not lotto_cols:
        st.error("Le fichier CSV doit contenir les colonnes N1, N2, N3, N4, N5, N6")
        st.stop()

    # Flatten all numbers into a single series
    all_numbers = df[lotto_cols].values.flatten()

    # Compute frequency counts, sort by number ascending
    freq = pd.Series(all_numbers).value_counts().sort_index()

    st.subheader("Fréquence des numéros tirés")
    st.write(freq.to_frame(name="Fréquence"))

    # Plot frequency bar chart with legend
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(freq.index, freq.values, color='dodgerblue', label='Nombre de tirages')
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Fréquence")
    ax.set_title("Fréquence des numéros dans les tirages importés")
    ax.legend()

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    st.pyplot(fig)
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
