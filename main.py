from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph, START
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from helper_functions import collect_unique_values
from tools import create_tools
from sys_prompt import plot_prompt, router_prompt, answer_prompt
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_sandbox import PyodideSandbox
from langchain_core.messages import ToolMessage
load_dotenv()

class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        user_query : The user's query.
        result : The result of processing the query.
        node_name : The name of the node that processed the query.
    """
    user_query: str
    result: str
    node_name: str

class Plot(BaseModel):
    """The Chart Plotting Agent.Use Plot when asked to make a chart or a graph."""

class Answer(BaseModel):
    """The Question Answering Agent. Use when only required to answer questions and not generate plots."""

async def main(db_path=None, user_query=None):
    # Initialize DB, LLM, and tools
    db_uri = f"sqlite:///{db_path}" if db_path else "sqlite:///Chinook.db"
    db = SQLDatabase.from_uri(db_uri)
    llm = ChatGroq(model="qwen/qwen3-32b",temperature=0,reasoning_format="hidden")
    '''llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    include_reasoning=False,
)'''
    tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    if len(vector_store.get()["documents"]) != 0:
        vector_store.reset_collection()

    unique_values = collect_unique_values(db, strip_numbers=True)
    vector_store.add_texts(unique_values)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    tools += tuple(create_tools(retriever))
    plot_tools = [tool for tool in tools if tool!="sandbox_tool"]
    # Agent
    answer_agent = create_react_agent(model=llm, tools=tools, prompt=answer_prompt)
    # Create a global sandbox instance
    sandbox = PyodideSandbox(allow_net=True)
    # Define the Plot node
    async def Plot(state: GraphState) -> GraphState:
        try:
            question = state["user_query"]
            state["node_name"] = "Plot"
            sql_result=None

            for chunk in answer_agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values", 
    ):
                msg = chunk["messages"][-1]
                content = msg.content
                if content and isinstance(msg,ToolMessage) and  msg.name=="sql_db_query":
                    sql_result = eval(content)

            async def plot_agent(question,sql_result):
                chart_input = f"{plot_prompt}\n\nUser Question: {question}\nDataFrame: {sql_result}"
                result = await llm.ainvoke(chart_input)
                return result.content
            code = await (plot_agent(question=question,sql_result=sql_result))
            exec_result = await sandbox.execute(code)
            state["result"] = exec_result.result

        except Exception:
            state["result"] = None  # Fallback

        return state


    # Define the Answer node
    async def Answer(state: GraphState) -> GraphState:
        try:
            question = state["user_query"]
            state["node_name"] = "Answer"
            result = await answer_agent.ainvoke({"messages": [{"role": "user", "content": question}]})
            if result.get("messages") and result["messages"][-1].content:
                content = result["messages"][-1].content
            else:
                content = "No response from Answer agent."
            state["result"] = content
        except Exception:
            state["result"] = "An error occurred in Answer."
        return state

    #llm router
    structured_llm_router = llm.bind_tools(
        tools=[Plot, Answer],
    )

    route_prompt = ChatPromptTemplate.from_messages([
        ("system",router_prompt),
        ("human", "{user_query}")
    ])
    question_router = route_prompt | structured_llm_router

    #defining route
    async def route(state: GraphState) -> str:
        """Route to Plot or Answer."""
        question = state["user_query"]
        result = await question_router.ainvoke({"user_query": question})
        try:
            tool_call = result.tool_calls[0]["name"]
        except Exception:
            tool_call = "Answer"

        return tool_call if tool_call in ["Plot", "Answer"] else "Answer"


    if not user_query:
        return

    # Define the state graph
    workflow = StateGraph(GraphState)

    workflow.add_node("Plot", Plot)
    workflow.add_node("Answer", Answer)
    workflow.add_conditional_edges(
        START,
        route,
        {
            "Plot": "Plot",
            "Answer": "Answer",
        }
    )

    workflow.add_edge("Plot", END)
    workflow.add_edge("Answer", END)

    # Compile and execute the graph
    app = workflow.compile()
    result_state = await app.ainvoke({"user_query": user_query})
# Return both the result and the node used
    return {"result": result_state["result"], "node": result_state.get("node_name", "")}


