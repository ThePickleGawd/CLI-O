import json
import re
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from rag import *


github_url = "https://github.com/ThePickleGawd/geometry-dash-ai"

query_engine = setup_query_engine(github_url=github_url)

retriever = query_engine.retriever
retriever.similarity_top_k = 4

def github_rag_tool(query:str) -> str:
    try:
        nodes = retriever.retrieve(query)
        context = "\n\n".join([node.node.get_content() for node in nodes])
        return context  
    except Exception as e:
        return f"[GithubRepoRAG Error]: {str(e)}"
    

GithubRepoRAG = Tool(
    name="GithubRepoRAG",
    description="Use this tool to ask questions about the Github repository",
    func=github_rag_tool
)



tools = {
    "duckduckgo_search": DuckDuckGoSearchRun(),
    "wikipedia": WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    "python": PythonREPLTool(),
    "GithubRepoRAG": GithubRepoRAG
}


llm = ChatOllama(model="mistral", temperature=0.2)

system_prompt = """
You are a helpful AI agent. You have access to the following tools:

duckduckgo_search: use this to search the internet
wikipedia: use this to look up encyclopedia information
python: use this to do calculations or run code
GithubRepoRAG: use this to query local code repositories and understand context of what's happening in them

Only use tools when absolutely necessary. If you already know the answer, respond directly without using tools.

When you want to use a tool, respond with a JSON object like this:

{
  "action": "tool_name",
  "action_input": "input to the tool"
}

Only respond with that JSON. Once you receive the tool result, incorporate it into your final answer. Keep it concise and to the point.
"""


def run_custom_agent(user_input):
    prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
    response = llm.invoke(prompt).content.strip()

    print(f"\n🔹 LLM Raw Output:\n{response}\n")

    # Try to parse action block
    try:
        # Use regex to find JSON blob
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if not json_match:
            raise ValueError("No JSON found.")
        action_blob = json.loads(json_match.group())

        action = action_blob.get("action")
        action_input = action_blob.get("action_input")

        if action == "none" or action is None:
            return response.replace(str(action_blob), "").strip()

        if action not in tools:
            return f"Unknown tool: {action}"

        print(f"⚙️ Running Tool: {action}({action_input})")
        result = tools[action].run(action_input)

        # Feed result back into LLM
        followup_prompt = (
            f"{system_prompt}\n\nUser: {user_input}\n"
            f"Tool [{action}] returned:\n{result}\n"
            f"Assistant: Now write a final answer using the tool result."
        )
        final_response = llm.invoke(followup_prompt).content.strip()
        return final_response

    except Exception as e:
        return f"[Error parsing tool call]: {e}\n\nRaw output:\n{response}"


if __name__ == "__main__":
    user_input = "How does this local repo use reinforcement learning?"
    final_response = run_custom_agent(user_input)
    print(f"\n Final Answer:\n{final_response}")