from langgraph_agent.nodes import llm_node
from langgraph_agent.graph import AgentState, build_graph
from textwrap import dedent

# Create the graph
agent = build_graph()

# Create a state graph
text = dedent("""\
    If I have these results for a regression, build a summary of how I can improve my model.
        Be concise and avoid unnecessary details.
        Give me actionable suggestions.
              
        Results:
        R-squared: 0.44
        RMSE: 0.84
        Intercept: 0.45

        Coefficients:
               feature  coefficient
        0   total_bill     0.094700
        1         size     0.233484
        2     sex_Male     0.028819
        3    smoker_No     0.192353
        4      day_Sat    -0.006064
        5      day_Fri     0.179721
        6      day_Sun     0.128928
        7  time_Dinner    -0.094957


        VIF:
        total_bill    2.226294
        tip           1.879238
        size          1.590524
              \
""")
prompt = {
    "messages": [],
    "question": text
}

# Result
result = agent.invoke(prompt)

# Print the agent's response
for m in result["messages"]:
    m.pretty_print()