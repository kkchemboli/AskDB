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
    llm = ChatOllama(model="mistral", temperature=0)

    system_message = """
You are an agent designed to interact with a {dialect} database using tools. 

## Rules for Behavior
1. You MUST always start by calling the `list_tables` tool to see which tables exist. 
   - Do not guess table names.
2. After identifying candidate tables, you MUST call the `tables_schema` tool 
   to inspect their schema before writing any query.
3. Before executing a SQL query, you MUST call the `check_sql` tool to validate it.
4. Only then should you call `execute_sql` to run the query.
5. If a query fails, rewrite and retry until it executes successfully.
6. Never return pseudo-code or describe tool usage in natural language — always invoke the actual tools.
7. Never make DML (INSERT, UPDATE, DELETE, DROP, etc.) statements. You are read-only.

## SQL Guidelines
- You are writing {dialect} queries.
- Only query the relevant columns, never use `SELECT *`.
- Unless the user specifies otherwise, limit results to at most {top_k}.
- You can order results by a relevant column to show the most useful data.
- Double check your SQL carefully before execution.

## Goal
Given a user’s question, use the tools step by step to find the correct answer from the database.
""".format(
    dialect="SQLite",
    top_k=5
)

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

    question = "List all the tables in the database."

    for step in agent.stream({"messages": [{"role": "user", "content": question}]},stream_mode="values",):
     step["messages"][-1].pretty_print()




if __name__ == "__main__":
    main()
