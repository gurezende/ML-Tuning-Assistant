from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph_agent.nodes import llm_node


# Create a state graph
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages in the conversation, including user input and agent outputs.
        question: The initial question from the user.
    """
    messages: Annotated[AnyMessage, add_messages] # accumulate messages
    question: str


def build_graph():
    # Build the LangGraph flow
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("llm", llm_node)

    # Define edges and flow
    builder.set_entry_point("llm")
    builder.add_edge("llm", END)

    # Compile the graph
    return builder.compile()


# Create the graph image and save png
# from IPython.display import display, Image
# graph = build_graph()
# display(Image(graph.get_graph().draw_mermaid_png()))