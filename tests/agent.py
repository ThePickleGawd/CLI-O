from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, KokoroEngine
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from pydantic import PrivateAttr
import torch

# ================== LangChain Tools ===================
tools = [
    # CalculatorTool(),
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    PythonREPLTool()
]

# ================== Qwen LangChain Wrapper ===============
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
        return self  # Stub needed for LangChain agent support

    @property
    def _llm_type(self) -> str:
        return "qwen"
    

# =================== Model Setup ========================
print("Loading Qwen model...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
llm = QwenLangChainWrapper(tokenizer, model)

# =================== Agent Setup ========================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful voice assistant. Use tools when needed, and only respond to the user's most recent question. Do not continue the conversation, ask follow-up questions, or generate multiple turns. Respond clearly and concisely."
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# =================== TTS Setup ===========================
print("Initializing TTS...")
engine = KokoroEngine()
stream = TextToAudioStream(engine, frames_per_buffer=256)

# =================== Text Processing =====================
def process_text(text):
    print(f"> You said: {text}")
    result = agent_executor.invoke({"input": text})
    raw_output = result["output"]

    # Strip system/user/assistant prefixes and any [END_OF_TEXT]
    lines = raw_output.splitlines()
    cleaned_lines = [
        line for line in lines
        if not line.strip().startswith("[SYSTEM]")
        and not line.strip().startswith("[USER]")
        and not line.strip().startswith("[ASSISTANT]")
        and "[END_OF_TEXT]" not in line
    ]
    cleaned_output = " ".join(cleaned_lines).strip()
    print(f"> Agent: {cleaned_output}")

    def generator():
        yield from cleaned_output

    stream.feed(generator())
    stream.play(log_synthesized_text=True)

# =================== Main Loop ===========================
if __name__ == '__main__':
    print("Say something like 'What is 8 * 7?' or 'Search Wikipedia for Ada Lovelace'")
    # recorder = AudioToTextRecorder(
    #     enable_realtime_transcription=True,
    #     silero_use_onnx=True,
    #     no_log_file=True,
    # )

    # while True:
    #     text = recorder.text()
    #     process_text(text)
    text = "Who was the 5th president of the United States?"
    process_text(text)
