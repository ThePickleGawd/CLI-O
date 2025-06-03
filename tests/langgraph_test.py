import json
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

from rag import *

github_url = "https://github.com/ThePickleGawd/geometry-dash-ai"

query_engine = setup_query_engine(github_url=github_url)

retriever = query_engine.retriever
retriever.similarity_top_k = 4

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

class ToolUse(BaseModel):
    tool: Literal["RAGTool", "NoTool", "WikipediaTool"] = Field(
        description="Decide whether to use any given tools or no tool at all if you feel none is necessary. " \
        "The RAGTool allows you to query a local repo to get understanding of what is happening with the code" \
        "Type NoTool if you feel no tool is necessary"
    )

tool_caller = llm.with_structured_output(ToolUse)

class State(TypedDict):
    input: str
    tool: str
    additionalContext: str
    output: str

def RAGToolCall(state: State):
    try:
        nodes = retriever.retrieve(state["input"])
        context = "\n\n".join([node.node.get_content() for node in nodes])
        return {"additionalContext": context}  
    except Exception as e:
        return {"additionalContext": f"[GithubRAGToolError] : {str(e)}"}
    
def NoToolCall(state: State):
    return {"additionalContext": "No additional context needed"}

def WikipediaToolCall(state: State):
    try:
        query=state["input"]
        result=wiki_tool.run(query)
        return {"additionalContext": result}
    except Exception as e:
        return {"additionalContext": f"[WikipediaToolError] : {str(e)}"}
    
def llm_call_router(state: State):
    decision = tool_caller.invoke(
        [
            SystemMessage(
                content="Decide whether to use any given tools or no tool at all if you feel none is necessary. " \
                "The RAGTool allows you to query a local repo to get understanding of what is happening with the code" \
                "The WikipediaTool allows you to search wikipedia for any information you need to know" \
                "Type NoTool if you feel no tool is necessary"
            ),
            HumanMessage(content=state['input'])
        ]
    )
    return {"tool": decision.tool} 

def synthesizer(state: State):
    return 
    final_output = llm.invoke(
        [
            SystemMessage(content="Write a final answer given the query and additional context. Make sure the response in concise. 2-3 sentences Maximum"),
            HumanMessage(content=f"Query: {state['input']} \n Context: {state['additionalContext']}")
        ]
    )
    return {"output": final_output}

def route_decision(state: State):
    if state["tool"] == "RAGTool":
        return "RAGTool"
    elif state["tool"] == "NoTool":
        return "NoTool"
    elif state["tool"] == "WikipediaTool":
        return "WikipediaTool"
    
    
router_builder = StateGraph(State)

router_builder.add_node("RAGToolCall", RAGToolCall)
router_builder.add_node("NoToolCall", NoToolCall)
router_builder.add_node("WikipediaToolCall", WikipediaToolCall)
router_builder.add_node("llm_call_router", llm_call_router)
#router_builder.add_node("synthesizer", synthesizer)


router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router", 
    route_decision, 
    {
        "RAGTool": "RAGToolCall",
        "NoTool": "NoToolCall",
        "WikipediaTool": "WikipediaToolCall"
    }
)
router_builder.add_edge("RAGToolCall", END)
router_builder.add_edge("NoToolCall", END)
router_builder.add_edge("WikipediaToolCall", END)
#router_builder.add_edge("synthesizer", END)

router_workflow = router_builder.compile()
output = router_workflow.invoke({"input":"Who is Ada Lovelace?"})

print(output["additionalContext"])


def run_agent(input: str):
    output = router_workflow.invoke({"input": input})
    return output["additionalContext"]



    