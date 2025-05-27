from tools import tool_instances
from typing import TypedDict, Literal, Any, Dict

class FunctionCall(TypedDict):
    name: str
    arguments: Dict[str, Any]

def run_tool_call(tool_call: FunctionCall):
    name = tool_call["name"]
    args = tool_call["arguments"]
    
    tool = tool_instances.get(name)

    if tool is None:
        raise ValueError(f"Tool '{name}' not found.")

    return tool(**args)