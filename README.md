# Local Realtime Speech Agents

A fully local speech-to-speech AI pipeline combining real-time speech recognition, LLM reasoning, tool augmentation, and low-latency text-to-speech—built to run entirely on-device for faster, private, and extensible interactions.

[Paper](/docs/paper.pdf) | [Demo](https://drive.google.com/file/d/1JloowwSbQ0DcNZWMM6BvuEYxB-4Mc93c/view?usp=drive_link)

### Pipeline

```
User Speech → STT → LLM (Tool-Augmented) → TTS → Audio Response
```

- **STT**: Converts audio into text (e.g., [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT))
- **LLM + Tools**: Text input is processed by a local LLM (in `models/`) which can invoke external tools (from `tools/`)
- **TTS**: Streams the LLM response as audio (e.g., [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS))

![Models](/docs/models.png)

## Setup

```bash
# Install uv
pip install uv

git clone https://github.com/ThePickleGawd/realtime-speech-agents.git
cd realtime-speech-agents
uv sync
```

## Models

- `V1.py`, `V2.py`, `V3.py`: Variants of the core speech agent models.
- Includes LLM orchestration logic, response synthesis, and tool invocation.
- See paper for more details

## ▶️ Running the Agent

```bash
uv run models/V1.py
```

Speak into the mic — your agent will respond in real time.
