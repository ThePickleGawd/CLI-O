from RealtimeTTS import TextToAudioStream, KokoroEngine
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
import time
import json
from questions import questions
from langgraph_test import run_agent

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
messages = [{"role": "system", "content": system_prompt}]

def save_results(responses, times, output_file="results_langgraph.json"):
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
        start = time.time()
        response = run_agent(question)
        print(f"Response: {response}")
        time_taken = time.time() - start

        responses[question] = response
        times[question] = time_taken

        print(f"Time taken: {time_taken:.2f} seconds")
        save_results(responses, times)

    print("\nAll questions processed! Results saved to results_langgraph.json")
