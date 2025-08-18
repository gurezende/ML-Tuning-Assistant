
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph_agent.nodes import llm_node_classification, llm_node_regression, get_model_type


# Create a state graph
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages in the conversation, including user input and agent outputs.
        model_type: The type of model being used, either "classification" or "regression".
        question: The initial question from the user.
        final_answer: The final answer provided by the agent.
    """
    messages: Annotated[AnyMessage, add_messages] # accumulate messages
    model_type: str
    metrics_to_tune: str
    final_answer: str


def build_graph():
    # Build the LangGraph flow
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("get_model_type", get_model_type)
    builder.add_node("classification", llm_node_classification)
    builder.add_node("regression", llm_node_regression)

    # Define edges and flow
    builder.set_entry_point("get_model_type")

    builder.add_conditional_edges(
        "get_model_type",
        lambda state: state["model_type"],
        {
            "classification": "classification",
            "regression": "regression"
        }
    )

    builder.add_edge("classification", END)
    builder.add_edge("regression", END)

    # Compile the graph
    return builder.compile()


# Create the graph image and save png
# from IPython.display import display, Image
# graph = build_graph()
# display(Image(graph.get_graph().draw_mermaid_png()))
