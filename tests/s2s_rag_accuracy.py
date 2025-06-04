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

def save_results(responses, output_file="results_langgraph.json"):
    results = {
        "responses": responses
    }
    with open(output_file, 'a') as f:
        json.dump(results, f, indent=2)

# ================ Main Processing Loop ==============
if __name__ == '__main__':
    print("Starting to process Geometry Dash questions...")

    responses = {}

    gd_questions = questions[-20:]
    gd_questions = [f"Using the local geometry dash repo, {q}" for q in gd_questions]

    for i, question in enumerate(gd_questions, 1):
        print(f"\nProcessing question {i}/{len(gd_questions)}: {question}")
        response = run_agent(question)
        print(f"Response: {response}")

        responses[question] = response
        save_results(responses)

    print("\nAll Geometry Dash questions processed! Results saved to results_langgraph.json")

