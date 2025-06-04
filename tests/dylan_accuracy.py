from RealtimeTTS import TextToAudioStream, KokoroEngine
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading
import time
import json
from questions import questions
import threading
from utils.parse import try_parse_tool_calls
from tools import tools
from tools.functions import run_tool_call
from datetime import datetime

# ============== Init TTS and Warmup =================
print("Initializing TTS system...")
engine = KokoroEngine()
stream = TextToAudioStream(engine, frames_per_buffer=256)
stream.feed("Warmup complete")
stream.play(muted=True)

# =================== LLM Setup =====================
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
system_prompt = f"""You are a helpful assistant who responds naturally, like a real person speaking out loud. Start with short, clear sentences to reduce delay in speech. Avoid robotic or overly formal language. If a tool call is needed, say which tool you will call, then immediately call it. If you don't know something, look it up. Today is: {datetime.today().strftime("%Y-%m-%d")}"""
blank_message = [
    {"role": "system", "content": system_prompt}
]

# Reset every test
messages = blank_message.copy()

def process_text(text=None):

    if text != None:
        messages.append({"role": "user", "content": text})

    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools
    )
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate():
        model.generate(**model_inputs, streamer=streamer, max_new_tokens=128)

    # Start generation thread
    gen_thread = threading.Thread(target=generate)
    gen_thread.start()

    generate_again = False
    full_response = ""
    first_token_time = None
    for token in streamer:
        full_response += token

        if first_token_time is None:
            first_token_time = time.time()
        
        # Don't speak anything related to tool call. Kinda hacky
        if "<tool_call>" in full_response and "</tool_call>" not in full_response and token != "</tool_call>":
            continue

    result = try_parse_tool_calls(full_response)
    
    if "tool_calls" in result:
        # Append assistant message with tool_calls but empty content
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": result["tool_calls"]
        })
        for tool_call in result["tool_calls"]:
            res = run_tool_call(tool_call["function"])
            tool_message = {
                "role": "tool",
                "name": tool_call["function"]["name"],
                "content": res
            }
            messages.append(tool_message)

        generate_again = True
    else:
        # No tool calls; just append the assistant response as usual
        messages.append({
            "role": "assistant",
            "content": full_response
        })
    
    return first_token_time, generate_again

# =================== LLM Setup =====================
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

def save_results(responses, times, output_file="results_dylan.json"):
    total_time = sum(times.values())
    avg_time = total_time / len(times) if times else 0.0
    results = {
        "responses": responses,
        "timing": {
            "individual": times,
            "total_questions": len(responses),
            "total_time": total_time,
            "average_time": avg_time
        }
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

# ================ Main Processing Loop ==============
if __name__ == '__main__':
    print("Starting to process questions...")
    responses = {}
    times = {}

    for i, question in enumerate(questions, 1):
        print(f"\nProcessing question {i}/{len(questions)}: {question}")
        messages = blank_message.copy()
        start = time.time()
        first_token_time, generate_again = process_text(question)

        if generate_again:
            generate_again = False
            process_text(None)

        response = "".join([m["content"] for m in messages if m["role"] == "assistant"])
        print(f"Response: {response}")
        time_taken = time.time() - start

        responses[question] = response
        times[question] = time_taken

        print(f"Time taken: {time_taken:.2f} seconds")
        save_results(responses, times)

    print("\nAll questions processed! Results saved to results_dylan.json")
