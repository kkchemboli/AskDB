import streamlit as st
import os
import tempfile
import streamlit.components.v1 as components
import asyncio
from helper_functions import save_if_html
from main import main as agent_main
import base64

st.title("AskDB: Natural Language to SQL & Plotly Charts")

# Step 1: Upload SQLite DB
db_file = st.file_uploader("Upload a SQLite .db file", type=["db"])
db_path = None
if db_file is not None:
    temp_dir = tempfile.gettempdir()
    db_path = os.path.join(temp_dir, db_file.name)
    with open(db_path, "wb") as f:
        f.write(db_file.getbuffer())
    st.success(f"Database uploaded: {db_file.name}")

# Step 2: Enter Query
query = st.text_area("Enter your query (e.g., 'Plot a pie chart showing the distribution of tracks by genre.')")

# Step 3: Run Agent and Display Output
if st.button("Submit Query") and db_path and query:
    
    # Show spinner while generating plot
    with st.spinner("Generating response... Please wait."):
        
        answer =  asyncio.run(agent_main(db_path, query))

        if answer["node"] == "Plot":
            img_bytes = base64.b64decode(answer["result"])
            st.image(img_bytes)
        else:
            st.write(answer["result"])
    
    

    
