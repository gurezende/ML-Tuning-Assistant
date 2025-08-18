import os
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_model_type(state):
    """Check if the user is implementing a classification or regression model. """
    
    # Define the model type
    modeltype = st.text_input("Please let me know the type of model you are working on and hit Enter:", 
                              placeholder="(C)lassification or (R)egression", 
                              help="C for Classification or R for Regression")
    
    # Check if the model type is valid
    if modeltype.lower() not in ["c", "r", "classification", "regression"]:
        st.info("Please enter a valid model type: C for (C)lassification or R for (R)egression.")
        st.stop()
        
    if modeltype.lower() in ["c", "classification"]:
        modeltype = "classification"
    elif modeltype.lower() in ["r", "regression"]:
        modeltype = "regression"
    
    return {"model_type": modeltype.lower()} # "classification" or "regression"  


def llm_node_classification(state):
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
        ("system", dedent("""\
                          You are a seasoned data scientist, specialized in classification models. 
                          You have a deep understanding of classification models and their applications.
                          You will get the user's result for a classification model and your task is to build a summary of how to improve the model.
                          Use the context to answer the question.
                          Give me actionable suggestions in the form of bullet points.
                          Be concise and avoid unnecessary details. 
                          If the question is not about classification, say 'Please input classification model metrics.'.
                          \
                          """)),
        MessagesPlaceholder(variable_name="messages"),
        ("user", state["metrics_to_tune"])
    ])
    
    # Create a chain
    chain = messages | llm
    response = chain.invoke(state)
    return {"final_answer": [response]}


def llm_node_regression(state):
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
        ("system", dedent("""\
                          You are a seasoned data scientist, specialized in regression models. 
                          You have a deep understanding of regression models and their applications.
                          You will get the user's result for a regression model and your task is to build a summary of how to improve the model.
                          Use the context to answer the question.
                          Give me actionable suggestions in the form of bullet points.
                          Be concise and avoid unnecessary details. 
                          If the question is not about regression, say 'Please input regression model metrics.'.
                          \
                          """)),
        MessagesPlaceholder(variable_name="messages"),
        ("user", state["metrics_to_tune"])
    ])
    
    # Create a chain
    chain = messages | llm
    response = chain.invoke(state)
    return {"final_answer": [response]}



