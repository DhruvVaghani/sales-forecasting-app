import streamlit as st
import sqlite3
import pandas as pd






st.set_page_config(page_title="Sales Forecasting Q&A", layout="wide")

st.title("📊 Sales Forecasting Assistant")
st.markdown("Ask natural language questions like:")
st.code("predict sales for store number 1, family EGGS on 17th August 2017")


# Load and display preview data
conn = sqlite3.connect("mydatabase (1).db")
df = pd.read_sql("SELECT * FROM forecasted_sales LIMIT 5", conn)
conn.close()
# Compute basic metadata
total_rows = len(df)
num_stores = df["store_nbr"].nunique()
num_families = df["family"].nunique()
min_date = pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d")
max_date = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")

# Summary info
st.markdown("### 📊 Forecasted Sales Database Summary")
st.markdown(f"""
- 🔢 **Total forecasts stored:** `{total_rows}`
- 🏬 **Number of stores:** `{num_stores}`
- 📦 **Number of product families:** `{num_families}`
- 📅 **Date range:** `{min_date}` to `{max_date}`
""")
st.subheader("📁 Preview of Forecasted Sales Data")
st.dataframe(df)

st.markdown("""
### 🧠 How to Use This Assistant
- The database includes forecasts for combinations of store number, family, and date.
- Use natural language like:
    - `"show predicted sales for store 3, family BREAD on 5 August 2018"`
    - `"forecast sales for store 5, family EGGS for 5th August 2026"`
- If the forecast doesn't exist, the assistant will generate it using a trained ML model.
""")

st.markdown("---")
st.subheader("🔍 Ask a Forecasting Question")

user_input = st.text_input("Type your question:", placeholder="e.g., predict sales for store number 2, family MILK on 2025-12-01")

from app import run_agent  # assumes both files are in same folder

if user_input:
    with st.spinner("Thinking..."):
        result = run_agent(user_input)

        st.success(result)



        
