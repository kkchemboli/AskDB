import streamlit as st
import os
import tempfile
import streamlit.components.v1 as components
import asyncio

st.title("LangGraphSQL: Natural Language to SQL & Plotly Charts")

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
    from test import save_if_html, main as agent_main

    # Show spinner while generating plot
    with st.spinner("Generating plot... Please wait."):
        async def run_agent_with_custom_input():
            await agent_main(db_path, query)
            # Find the latest temp HTML file
            temp_htmls = [os.path.join(tempfile.gettempdir(), f) for f in os.listdir(tempfile.gettempdir()) if f.endswith(".html")]
            if temp_htmls:
                latest_html = max(temp_htmls, key=os.path.getctime)
                with open(latest_html, "r", encoding="utf-8") as f:
                    html_content = f.read()
                return html_content, latest_html
            return None, None

        html_content, html_path = asyncio.run(run_agent_with_custom_input())

    if html_content and (html_content.startswith("<!DOCTYPE html>") or html_content.startswith("<html")):
        components.html(html_content, height=600, scrolling=True)
    elif html_content:
        st.write(html_content)
    else:
        st.warning("No chart or valid output generated.")