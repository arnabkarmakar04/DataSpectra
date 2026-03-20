import streamlit as st
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from utils.tools import *


class AgentState(TypedDict):
    messages: List[dict]
    response: str


def build_copilot(df):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=st.secrets["GOOGLE_API_KEY"]
    )

    pandas_tool = create_pandas_tool(df, model)
    trend_tool = create_trend_tool(df, model)

    system_prompt = f"""
You are a Data Analysis Copilot embedded in a data exploration system.

Your role is to answer user queries by leveraging specialized tools, ensuring all responses are accurate, data-driven, and reproducible.

━━━━━━━━━━━━━━━━━━━━━━━
CORE INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━
- You MUST use one of the available tools to answer every query.
- You MUST call a tool BEFORE providing any final answer.
- You are NOT allowed to answer from your own knowledge or assumptions.
- If you do not use a tool, your response is INVALID.

━━━━━━━━━━━━━━━━━━━━━━━
TOOL USAGE
━━━━━━━━━━━━━━━━━━━━━━━
- Use `pandas_query_tool` for structured data queries such as:
  filtering, aggregation, groupby, statistics, or general analysis.

- ALWAYS use `trend_analysis_tool` when the query involves:
  trends, patterns, increases, decreases, or time-based behavior.

- Do NOT write pandas code yourself.
- Always pass the user query to the appropriate tool.

━━━━━━━━━━━━━━━━━━━━━━━
YOUR RESPONSIBILITY
━━━━━━━━━━━━━━━━━━━━━━━
- Understand the user’s intent clearly.
- Select the correct tool based on the query type.
- Call the tool with the appropriate input.
- Interpret and explain the tool’s output in a clear, concise manner.

━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STYLE
━━━━━━━━━━━━━━━━━━━━━━━
- Be precise and structured.
- Do not include unnecessary explanations.
- Focus only on insights derived from the data.

━━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━
- Never modify the dataset.
- Never generate plots or visualizations.
- Never import libraries or execute external operations.
"""

    agent = create_agent(
        model=model,
        tools=[pandas_tool, trend_tool],
        system_prompt=system_prompt
    )

    def agent_node(state: AgentState):
        response = agent.invoke({
            "messages": state["messages"]
        })

        content = response["messages"][-1].content

        if isinstance(content, str):
            final = content
        elif isinstance(content, list):
            final = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        else:
            final = str(content)

        return {"response": final}

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    return graph.compile()