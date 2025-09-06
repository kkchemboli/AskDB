from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import create_react_agent
import asyncio
from langchain_chroma import Chroma
from helper_functions import save_if_html, collect_unique_values
from tools import create_tools

async def main(db_path=None, user_query=None):
    # Initialize DB, LLM, and tools
    db_uri = f"sqlite:///{db_path}" if db_path else "sqlite:///Chinook.db"
    db = SQLDatabase.from_uri(db_uri)
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

    tools += tuple(create_tools(retriever))

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
   - Always use the correct column names from the SQL results for the chart axes (x, y, etc.).
   - Label the axes and chart title using the actual column names and query context.
   - Convert the chart to HTML using fig.to_html().
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

    if not user_query:
        print("no query entered")
        return
    question = user_query

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        msg = chunk["messages"][-1]
        content = msg.content

        # Handle string or list output
        if isinstance(content, str):
            texts = [content.strip()]
        elif isinstance(content, list):
            texts = [b.get("text", "").strip() for b in content if isinstance(b, dict) and b.get("type") == "text"]
        else:
            texts = []

        for text in texts:
            if text:  # skip empty
                save_if_html(text, question)

if __name__ == "__main__":
    asyncio.run(main())