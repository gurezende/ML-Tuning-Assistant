import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def llm_node(state):
    """
    Processes the user query and search results using the LLM and returns an answer.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # Create a prompt
    messages = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the context to answer the question. If you don't know the answer, say I don't know."),
        MessagesPlaceholder(variable_name="messages"),
        ("user", state["question"])
    ])
    
    # Create a chain
    chain = messages | llm
    response = chain.invoke(state)
    return {"messages": [response]}







