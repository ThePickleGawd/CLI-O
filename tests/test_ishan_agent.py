import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool
from pydantic import PrivateAttr

from questions import questions  # expects a file questions.py with `questions = [...]`

# ============ Tools ================
tools = [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    PythonREPLTool(),
]

# ============ Qwen Wrapper =========
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
        outputs = self._model.generate(**inputs, max_new_tokens=256)
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return AIMessage(content=text.strip())

    def _call(self, messages, stop=None, **kwargs) -> str:
        prompt = self._convert_messages_to_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=256)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def bind_tools(self, tools):
        return self

    @property
    def _llm_type(self) -> str:
        return "qwen"

# ========== Load Model ============
print("Loading Qwen model...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()
llm = QwenLangChainWrapper(tokenizer, model)

# ========== Create Agent ===========
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who responds naturally, like a real person speaking out loud. Start with short, clear sentences to reduce delay in speech. Avoid robotic or overly formal language. Speak conversationally, as if you're talking to a friend. Keep your sentences concise, especially at the start of a response. Unless told otherwise, use shorter responses. Prioritize natural flow and clarity."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
agent = create_tool_calling_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ========== Run and Save ===========
output_path = Path("results_cot.json")
if output_path.exists():
    with output_path.open("r") as f:
        results = json.load(f)
else:
    results = {}

for idx, question in enumerate(questions):
    if str(idx) in results:
        continue  # Skip already answered

    print(f"\n🧠 Q{idx+1}: {question}")
    try:
        result = agent_executor.invoke({"input": question})
        results[str(idx)] = {
            "question": question,
            "output": result.get("output", "No output")
        }
    except Exception as e:
        results[str(idx)] = {
            "question": question,
            "error": str(e)
        }

    # Save after each question
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

print("✅ All questions evaluated and saved to results_cot.json.")
