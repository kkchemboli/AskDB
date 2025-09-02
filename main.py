from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

def main():
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    llm = ChatOllama(model="qwen2.5", temperature=0)

    system_message = """
You are an agent designed to interact with a {dialect} database using tools. 

## Rules for Behavior
1. You MUST always start by calling the `list_tables` tool for EVERY user question. 
   - Even if you think you already know the schema.
2. You MUST use `tables_schema` to confirm the columns before writing a query.
3. You MUST call `check_sql` before executing any SQL.
4. You MUST call `execute_sql` to get the answer.
5. You MUST NEVER respond with steps, explanations, or pseudo-code. 
6.Unless the user explicitly specifies otherwise, LIMIT all results to at most {top_k}.
   - If the user asks a question, you ONLY respond with final results.
""".format(dialect="SQLite",top_k=5)


    @tool("list_tables")
    def list_tables() -> str:
        """List the available tables in the database as a single comma-separated string."""
        return ListSQLDatabaseTool(db=db).invoke("")

    @tool("tables_schema")
    def tables_schema(tables: str) -> str:
        """
        Input is a comma-separated list of tables, output is the schema and sample rows
        for those tables. Be sure that the tables actually exist by calling `list_tables` first!
        Example Input: table1, table2, table3
        """
        tool = InfoSQLDatabaseTool(db=db)
        return tool.invoke(tables)

    @tool("check_sql")
    def check_sql(sql_query: str) -> str:
        """
        Use this tool to double check if your query is correct before executing it. Always use this
        tool before executing a query with `execute_sql`.
        """
        return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

    @tool("execute_sql")
    def execute_sql(sql_query: str) -> str:
        """Execute a SQL query against the database. Returns the result"""
        return QuerySQLDatabaseTool(db=db).invoke(sql_query)
    

    tools = [list_tables, tables_schema, check_sql, execute_sql]
    agent =  create_react_agent(model=llm,tools=tools,prompt=system_message)

    question = "Which genre has the highest average track length?"

    for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]}, 
    stream_mode="values"
):
        print(step["messages"][-1].pretty_print())




if __name__ == "__main__":
    main()
