import streamlit as st
import os
import tempfile
import asyncio
import sqlite3
import pandas as pd
from main import main as agent_main
import base64

st.title("AskDB: Natural Language to SQL & Visualizations")

# Step 1: Upload SQLite DB or CSV
db_file = st.file_uploader("Upload a SQLite .db file or a csv file", type=["db", "csv"])
db_path = None

if db_file is not None:
    temp_dir = tempfile.gettempdir()
    file_ext = os.path.splitext(db_file.name)[1].lower()
    db_path = os.path.join(temp_dir, db_file.name)

    with open(db_path, "wb") as f:
        f.write(db_file.getbuffer())

    if file_ext == ".csv":
        # Convert CSV to SQLite DB
        csv_db_path = os.path.join(temp_dir, db_file.name.replace(".csv", ".db"))
        df = pd.read_csv(db_path)
        table_name = "data"
        conn = sqlite3.connect(csv_db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        db_path = csv_db_path
        st.success(f"CSV uploaded and converted to SQLite DB: {os.path.basename(csv_db_path)}")
    else:
        st.success(f"Database uploaded: {db_file.name}")

# Step 2: Enter Query
query = st.text_area("Enter your query (e.g., 'Plot a pie chart showing the distribution of tracks by genre.')")

# Step 3: Run Agent and Display Output
if st.button("Submit Query") and db_path and query:
    with st.spinner("Generating response... Please wait."):
        answer = asyncio.run(agent_main(db_path, query))

        if answer["node"] == "Plot":
            img_bytes = base64.b64decode(answer["result"])
            st.image(img_bytes)
        elif answer["node"] == "Answer":
            st.write(answer["result"])
        else :
            st.write("No valid response generated.")




