from langchain_core.tools import tool
def create_tools(retriever):
    """
    Create and return all custom tools given a retriever.
    """

    description = (
        "Use to look up proper nouns. Input can be any approximate spelling, "
        "output is valid proper nouns from the database. Search is case-insensitive."
    )

    @tool("search_proper_nouns", description=description)
    def lowercase_query_wrapper(query: str):
        docs = retriever.invoke(query.lower())
        return [doc.page_content for doc in docs]


    # return tools 
    return [lowercase_query_wrapper]
