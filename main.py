from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.tools import tool
import re
from langchain_chroma import Chroma

# -------------------------
# Helper functions
# -------------------------
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
def main():
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
        # Convert user query to lowercase and return relevant documents
        docs = retriever.invoke(query.lower())
        # Return only the text content as a list of strings
        return [doc.page_content for doc in docs]

    tools.append(lowercase_query_wrapper)

    # System prompt for agent
    system_message = """
You are an agent designed to interact with a SQLite database using tools.

Rules:
1. Always call `list_tables` first.
2. Use `tables_schema` to confirm columns before writing a query.
3. Call `check_sql` before executing SQL.
4. Call `execute_sql` to get the answer.
5. Never respond with steps or explanations.
6. Limit all results to top 5 unless specified.
"""
    suffix = (
        "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
        "the filter value using the 'search_proper_nouns' tool! Do not try to "
        "guess at the proper name - use this function to find similar ones."
    )
    system = f"{system_message}\n\n{suffix}"

    # Create agent
    agent = create_react_agent(model=llm, tools=tools, prompt=system)

    # Example user query
    question = "How many albums does alis in chain have?"

    # Stream agent response
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values"
    ):
        print(step["messages"][-1].pretty_print())

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    main()
