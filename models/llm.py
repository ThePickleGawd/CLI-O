from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 

model_name = "Qwen/Qwen1.5-1.8B"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True).to('mps')
model.config.sliding_window = None


def respond_to_input(user_input, system_prompt = "You are a helpful assistant"):
    prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors ="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in full_response:
        return full_response.split("<|assistant|>")[-1].strip()
    else:
        return full_response.strip()
    
