from langgraph_agent.graph import AgentState, build_graph
from textwrap import dedent
import streamlit as st


## Config page
st.set_page_config(page_title="ML Model Tuning Assistant",
                   page_icon='🤖',
                   layout="wide",
                   initial_sidebar_state="expanded")


## SIDEBAR | Add a place to enter the API key
with st.sidebar:
    api_key = st.text_input("OPENAI_API_KEY", type="password")

    # Save the API key to the environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Clear
    if st.button('Clear'):
        st.rerun()
    
    st.divider()
    # About
    st.write("Designed with :heart: by [Gustavo R. Santos](https://gustavorsantos.me)")


## Title and Instructions
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    
st.title('ML Model Tuning Assistant | 🤖')
st.caption('This AI Agent is will help you tuning your machine learning model.')
st.write(':red[**1**] | 👨‍💻 Add the metrics of your ML model to be tuned in the text box. The more metrics you add, the better.')
st.write(':red[**2**] | ℹ️ Inform the AI Agent what type of model you are working on.')
st.write(':red[**3**] | 🤖 The AI Agent will respond with suggestions on how to improve your model.')
st.divider()

# Get the user input
text = st.text_area('**👨‍💻 Add here the metrics of your ML model to be tuned:**')

st.divider()


## Run the graph

# Spinner
with st.spinner("Gathering Tuning Suggestions...", show_time=True):
    from langgraph_agent.graph import build_graph
    agent = build_graph()

    # Create the initial state for the agent, with blank messages and the user input
    prompt = {
        "messages": [],
        "metrics_to_tune": text
    }

    # Invoke the agent
    result = agent.invoke(prompt)
    # Print the agent's response
    st.write('**🤖 Agent Response:**')
    st.write(result['final_answer'][0].content)
    



        
