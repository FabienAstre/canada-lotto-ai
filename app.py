import streamlit as st
import pandas as pd
from collections import Counter
import numpy as np

st.set_page_config(page_title="🎲 Canada Lotto 6/49", layout="wide")
st.title("🎲 Canada Lotto 6/49 Analyzer")

# ===========================
# 1️⃣ File Upload (top-level)
# ===========================
uploaded_file = st.file_uploader("Upload your Lotto 6/49 CSV", type=["csv"])

if not uploaded_file:
    st.warning("⚠️ Please upload your '649.csv' file to proceed.")
    st.stop()

# ===========================
# 2️⃣ Load CSV (cached)
# ===========================
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Ensure NUMBER DRAWN 1..6 columns exist
    required_cols = [f"NUMBER DRAWN {i}" for i in range(1, 7)]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"CSV missing required columns: {required_cols}")
    # Ensure numbers are integers
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
    # Optional: parse date
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
    return df

try:
    df = load_data(uploaded_file)
    st.success(f"Loaded {len(df)} historical draws.")
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

st.dataframe(df.tail(10))

# ===========================
# 3️⃣ Frequency Analysis
# ===========================
st.header("📊 Number Frequency")
all_numbers = df[[f"NUMBER DRAWN {i}" for i in range(1,7)]].values.flatten()
counter = Counter(all_numbers)
freq_df = pd.DataFrame(counter.items(), columns=["Number", "Frequency"]).sort_values(by="Frequency", ascending=False)
st.dataframe(freq_df)

# ===========================
# 4️⃣ Delta System (difference between numbers)
# ===========================
st.header("Δ Delta System")
def delta_system(row):
    numbers = sorted(row)
    deltas = [j-i for i, j in zip(numbers[:-1], numbers[1:])]
    return deltas

delta_df = df[[f"NUMBER DRAWN {i}" for i in range(1,7)]].apply(delta_system, axis=1)
st.dataframe(delta_df.tail(10))

# ===========================
# 5️⃣ Cluster / Zone Coverage
# ===========================
st.header("📍 Cluster / Zone Coverage")
# Example zones: 1-10,11-20,21-30,31-40,41-49
zones = [(1,10),(11,20),(21,30),(31,40),(41,49)]
def zone_coverage(row):
    counts = []
    for start,end in zones:
        counts.append(sum(start <= n <= end for n in row))
    return counts

zone_df = df[[f"NUMBER DRAWN {i}" for i in range(1,7)]].apply(zone_coverage, axis=1)
zone_df.columns = [f"Zone {i}" for i in range(1,6)]
st.dataframe(zone_df.tail(10))

# ===========================
# 6️⃣ Sum & Spread Filters
# ===========================
st.header("➗ Sum & Spread Filters")
sums = df[[f"NUMBER DRAWN {i}" for i in range(1,7)]].sum(axis=1)
spread = df[[f"NUMBER DRAWN {i}" for i in range(1,7)]].max(axis=1) - df[[f"NUMBER DRAWN {i}" for i in range(1,7)]].min(axis=1)
sum_spread_df = pd.DataFrame({"Sum": sums, "Spread": spread})
st.dataframe(sum_spread_df.tail(10))

# ===========================
# 7️⃣ Prediction Helper (Optional)
# ===========================
st.header("🔮 Prediction Helper")
most_common_numbers = freq_df.head(6)["Number"].tolist()
st.write(f"Top 6 frequent numbers (historical): {most_common_numbers}")

# ===========================
# End of App
# ===========================
st.success("✅ Lotto 6/49 Analysis Complete!")
