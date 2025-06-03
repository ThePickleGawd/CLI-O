from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, KokoroEngine
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
from time import sleep
import time
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

system_prompt = f"""You are a helpful assistant who responds naturally, like a real person speaking out loud. Start with short, clear sentences to reduce delay in speech. Avoid robotic or overly formal language. If a tool call is needed, say which tool you will call, then immediately call it. Today is: {datetime.today().strftime("%Y-%m-%d")}"""
messages = [
    {"role": "system", "content": system_prompt}
]

# Hacky setting
generate_again = False

def process_text(text=None):
    global messages, generate_again

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

    def generator():
        global generate_again

        full_response = ""
        for token in streamer:
            full_response += token
            
            # Don't speak anything related to tool call. Kinda hacky
            if "<tool_call>" in full_response and "</tool_call>" not in full_response and token != "</tool_call>":
                continue
            
            yield token

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


    # Process all test cases automatically
    print("Generating audio...")
    stream.feed(generator())
    stream.play(log_synthesized_text=True)

    print("\nAll generations completed!")


# ================ Main Audio Loop ==============
if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(
        enable_realtime_transcription=True, 
        silero_use_onnx=True,
        no_log_file=True,
    )

    while True:
        text = recorder.text()
        process_text(text)

        if generate_again:
            generate_again = False
            process_text(None)