import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import SimpleChatModel
from pydantic import PrivateAttr

# ==== Custom Tool Imports ====
from tools.duckduckgo import DDGSearch
from tools.wikipedia_summary import WikipediaSummary
from tools.python_intepreter import PythonInterpreter
from tools.linux_shell import LinuxShell


# ==== Qwen Wrapper for LangChain ====
class QwenLangChainWrapper(SimpleChatModel):
    _tokenizer: any = PrivateAttr()
    _model: any = PrivateAttr()

    def __init__(self, tokenizer, model):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model

    def _convert_messages_to_prompt(self, messages) -> str:
        prompt = ""
        for msg in messages:
            if msg.type == "system":
                prompt += f"[SYSTEM] {msg.content}\n"
            elif msg.type == "human":
                prompt += f"[USER] {msg.content}\n"
            elif msg.type == "ai":
                prompt += f"[ASSISTANT] {msg.content}\n"
        return prompt

    def predict_messages(self, messages, stop=None, **kwargs):
        prompt = self._convert_messages_to_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=512)
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return AIMessage(content=text.strip())

    def _call(self, messages, stop=None, **kwargs) -> str:
        prompt = self._convert_messages_to_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=512)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def bind_tools(self, tools):
        return self

    @property
    def _llm_type(self) -> str:
        return "qwen"


# ==== Load Qwen Model ====
print("Loading Qwen...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.eval()

llm = QwenLangChainWrapper(tokenizer, model)


# ==== Wrap Tools ====
def wrap_tool(name, description, func):
    return Tool(name=name, description=description, func=func)

search_tool = wrap_tool("search", "Search current information using DuckDuckGo.", DDGSearch(max_results=3))
wiki_tool = wrap_tool("wikipedia", "Get summaries from Wikipedia.", WikipediaSummary(max_results=3))
python_tool = wrap_tool("python", "Run Python code. Use print(...) to see output.", PythonInterpreter())
linux_tool = wrap_tool("linux", "Run Linux shell commands.", LinuxShell())

tools = [search_tool, wiki_tool, python_tool, linux_tool]


# ==== ReAct Prompt ====
prompt_template = PromptTemplate.from_template("""
You are a helpful voice assistant that reasons step by step and uses tools.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question
Thought: what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: a short, natural, spoken answer

Begin!

Question: {input}
{agent_scratchpad}
""")


# ==== Create Agent ====
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# ==== Run Example Questions ====
if __name__ == "__main__":
    questions = [
        "What is the square root of 144?",
        "When was Marie Curie born?",
        "What’s the weather like in San Francisco today?",
        "List files in the current folder.",
        "Generate 3 random numbers between 1 and 100 using Python.",
    ]

    for q in questions:
        print(f"\n🧠 Q: {q}")
        result = agent_executor.invoke({"input": q})
        print(f"✅ Final Answer: {result['output']}")
