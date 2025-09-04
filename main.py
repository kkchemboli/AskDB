from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.tools import tool
import re,asyncio,os
from langchain_chroma import Chroma
from langchain_sandbox import PyodideSandboxTool

# -------------------------
# Helper functions
# -------------------------
def save_if_html(text: str, query: str, graphs_dir: str):
    """Check if text is HTML and save, else print."""
    if text.startswith("<!DOCTYPE html>") or text.startswith("<html"):
        safe_query = re.sub(r"[^a-zA-Z0-9_-]+", "_", query.strip())[:80]
        filename = os.path.join(graphs_dir, f"{safe_query}.html")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Plotly chart saved to {filename}")
    else:
        print(text)
        

def get_text_columns(db, table):
    """Return all text-like columns in a table."""
    rows = db._execute(f"PRAGMA table_info({table});")
    return [
        r["name"] for r in rows
        if "CHAR" in r["type"].upper() or "TEXT" in r["type"].upper()
    ]

def collect_unique_values(db, strip_numbers=False):
    """
    Collect all unique proper nouns from all text columns in the database.
    Returns them as lowercase strings for case-insensitive matching.
    """
    all_values = set()
    tables = db.get_usable_table_names()

    for table in tables:
        text_cols = get_text_columns(db, table)
        for col in text_cols:
            rows = db._execute(f'SELECT DISTINCT "{col}" FROM "{table}"')
            for r in rows:
                v = r[col]
                if isinstance(v, str):
                    v = v.strip()
                    if strip_numbers:
                        v = re.sub(r"\b\d+\b", "", v).strip()
                    # ignore pure numbers
                    if not re.fullmatch(r"\d+(\.\d+)?", v):
                        # accept strings starting with any letter
                        if re.match(r"^[A-Za-z][\w\s&'-]+$", v):
                            all_values.add(v)

    return list(all_values)

# -------------------------
# Main function
# -------------------------
async def main():
    # Initialize DB, LLM, and tools
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    llm = ChatOllama(model="qwen2.5", temperature=0)
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

    # Initialize vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    if len(vector_store.get()["documents"]) == 0:
        unique_values = collect_unique_values(db, strip_numbers=True)
        vector_store.add_texts(unique_values)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    description = (
        "Use to look up proper nouns. Input can be any approximate spelling, "
        "output is valid proper nouns from the database. Search is case-insensitive."
    )

    @tool("search_proper_nouns", description=description)
    def lowercase_query_wrapper(query: str):
        docs = retriever.invoke(query.lower())
        return [doc.page_content for doc in docs]

    sandbox_tool = PyodideSandboxTool(allow_net=True)
    tools += (lowercase_query_wrapper, sandbox_tool)
    graphs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "graphs"))
    # System prompt
    system_message = f"""
You are an agent designed to interact with a SQLite database using tools.

Rules:
1. Always call `list_tables` first.
2. Use `tables_schema` to confirm columns before writing a query.
3. Call `check_sql` before executing SQL.
4. Call `execute_sql` to get the answer.
5. If the user asks for a chart, plot, or visualization:
   - Always generate Python code to create a pandas DataFrame from the SQL results.
   - Select an appropriate Plotly chart type.
   - Convert it to HTML using fig.to_html().
   - Return ONLY the raw HTML string that starts with <!DOCTYPE html>. 
   - Do NOT wrap it in Markdown fences (```) or add explanations, text, or formatting. 
   - Never include Python code in the final output — only the HTML string.
6. Never respond with steps or explanations.
7. Limit all results to top 5 unless specified.
"""
    suffix = (
        "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
        "the filter value using the 'search_proper_nouns' tool! Do not try to "
        "guess at the proper name - use this function to find similar ones."
    )
    system = f"{system_message}\n\n{suffix}"

    agent = create_react_agent(model=llm, tools=tools, prompt=system)

    question = "Plot a pie chart showing the distribution of tracks by genre."

    
    async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
        msg = chunk["messages"][-1]
        content = msg.content

        texts = []
        if isinstance(content, str):
            texts = [content.strip()]
        elif isinstance(content, list):
            texts = [b.get("text", "").strip() for b in content if isinstance(b, dict) and b.get("type") == "text"]

        for text in texts:
            if text:  # skip empty
                save_if_html(text, question, graphs_dir)






# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    os.makedirs("graphs", exist_ok=True)
    asyncio.run(main())