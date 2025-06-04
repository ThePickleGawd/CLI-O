from RealtimeTTS import TextToAudioStream, KokoroEngine
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
from time import sleep
import time
import threading
import json
from pathlib import Path
from collections import defaultdict

from langgraph_test import run_agent
from questions import questions

# ============== Init TTS and Warmup =================
print("Initializing TTS system...")
engine = KokoroEngine()
stream = TextToAudioStream(engine, frames_per_buffer=256)

# =================== LLM Setup =====================
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

system_prompt = """You are a helpful assistant who responds naturally, like a real person speaking out loud. Start with short, clear sentences to reduce delay in speech. Avoid robotic or overly formal language. Speak conversationally, as if you're talking to a friend. Keep your sentences concise, especially at the start of a response. Unless told otherwise, use shorter responses. Prioritize natural flow and clarity."""
messages = [
    {"role": "system", "content": system_prompt}
]

# Thread & streamer global
gen_thread = None
streamer = None
first = True

def process_text(text):
    global gen_thread, streamer, messages, first
    
    context = run_agent(text)

    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": context})

    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate():
        model.generate(**model_inputs, streamer=streamer, max_new_tokens=128)

    # Start generation thread
    gen_thread = threading.Thread(target=generate)
    gen_thread.start()

    def generator():
        global first
        full_response = ""
        for token in streamer:
            if first:
                first = False
                print(time.time() - start)
            full_response += token
            yield token
        messages.append({"role": "assistant", "content": full_response})
        return full_response

    # Process all test cases automatically
    print(f"Processing question: {text}")
    response = "".join(list(generator()))
    return response

def save_results(responses, times, output_file="results_langgraph.json"):
    # Calculate averages by category
    category_times = defaultdict(list)
    for question, time_taken in times.items():
        # Determine category from the question
        if "Use web search" in question:
            category = "web_search"
        elif "Use Wikipedia" in question:
            category = "wikipedia"
        elif any(keyword in question.lower() for keyword in ["plot", "calculate", "generate", "convert", "sort"]):
            category = "python"
        elif any(keyword in question.lower() for keyword in ["list", "show", "find", "copy", "move", "delete"]):
            category = "linux"
        else:
            category = "other"
        
        category_times[category].append(time_taken)
    
    # Calculate averages
    averages = {
        category: sum(times) / len(times) 
        for category, times in category_times.items()
    }
    
    # Prepare final results
    results = {
        "responses": responses,
        "timing": {
            "individual": times,
            "averages": averages,
            "total_questions": len(questions),
            "total_time": sum(times.values())
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
        start = time.time()
        response = process_text(question)
        print(f"Response: {response}")
        time_taken = time.time() - start
        
        responses[question] = response
        times[question] = time_taken
        
        print(f"Time taken: {time_taken:.2f} seconds")
        
        # Save after each question in case of interruption
        save_results(responses, times)
        
    print("\nAll questions processed! Results saved to results.json") 