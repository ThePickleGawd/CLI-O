import re, json, time, threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from utils.parse import try_parse_tool_calls
from tools.python_intepreter import PythonInterpreter

# ---- Model Setup ----
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.eval()

prompt = "Write and run a Python program that prints 'Hello, world!'. You can use a Python interpreter."

tools = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Run Python code to perform calculations or process data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python code to execute."}
                },
                "required": ["code"]
            }
        }
    }
]


messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools,
    tool_choice="auto"
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
thread = threading.Thread(
    target=model.generate,
    kwargs=dict(**model_inputs, streamer=streamer, max_new_tokens=512)
)
thread.start()

interpreter = PythonInterpreter()

# ---- Streaming + Tool Parse ----
buffer = ""
for token in streamer:
    print(token, end="", flush=True)
    buffer += token

    result = try_parse_tool_calls(buffer)
    if "tool_calls" in result:
        print("\n\n🔧 Parsed tool call:\n", result["tool_calls"])
        for tool_call in result["tool_calls"]:
            res = interpreter(tool_call["function"]["arguments"]["code"])
            print(f"Tool Result:\n{res}")

