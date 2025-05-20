from speech2text import *
from llm import * 

audio = record_audio()
temp_file = save_temp_audio(audio)
text = transcribe_with_whisper(temp_file)

print(text)

response = respond_to_input(text)

print(response)
