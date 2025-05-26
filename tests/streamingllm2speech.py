import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm

from RealtimeTTS import TextToAudioStream, KokoroEngine


@torch.no_grad()
def greedy_generate_stream(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0

    for _ in range(max_gen_len - 1):
        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().split(" ")
        now = len(generated_text) - 1
        if now > pos:
            for word in generated_text[pos:now]:
                print(word, end=" ", flush=True)
                yield word + " "
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break

    for word in generated_text[pos:]:
        print(word, end=" ", flush=True)
        yield word + " "
    print(flush=True)


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    # Initialize TTS
    engine = KokoroEngine()
    stream = TextToAudioStream(engine)
    print("Warming up TTS...")
    stream.feed("System initialized.")
    stream.play(muted=True)

    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt_text = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt_text, end="")
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

        if kv_cache is not None:
            space_needed = input_ids.shape[1] + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        generator = greedy_generate_stream(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )
        stream.feed(generator)
        stream.play(log_synthesized_text=True)


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(model, tokenizer, prompts, kv_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
