from tools.duckduckgo import DDGSearch
from tools.linux_shell import LinuxShell
from tools.python_intepreter import PythonInterpreter
from tools.wikipedia_summary import WikipediaSummary

tool_instances = {
    "ddg_search": DDGSearch(),
    "linux_shell": LinuxShell(),
    "python_interpreter": PythonInterpreter(),
    "wikipedia_summary": WikipediaSummary(),
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Run Python code to perform calculations or process data. "
                           "Use this for tasks like math, data parsing, or generating structured outputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Use `print(...)` to see output."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ddg_search",
            "description": "Use DuckDuckGo to search for general information or answers to questions. "
                           "Useful for looking up facts, people, definitions, or quick explanations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A search query such as 'What is quantum computing?' or 'Alan Turing'."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "linux_shell",
            "description": "Execute shell commands in a Linux environment. "
                           "Useful for file operations, system introspection, and scripting. "
                           "You can chain commands with semicolons. Install packages via `apk add ...`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The Linux shell command(s) to execute. Example: 'ls -la; pwd'"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_summary",
            "description": "Retrieve an encyclopedia-style summary of a topic from Wikipedia. "
                           "Use this for detailed, fact-checked background information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A topic or question like 'Marie Curie' or 'History of AI'."
                    }
                },
                "required": ["query"]
            }
        }
    }
]