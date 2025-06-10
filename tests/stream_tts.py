
from RealtimeTTS import TextToAudioStream, KokoroEngine
from time import sleep, time

def create_generator(text):
    """Create a text generator for a single text string"""
    def generator():
        for token in text.split():
            yield token + " "
            sleep(0.1)
    return generator()

def main():
    print("Initializing TTS system...")
    engine = KokoroEngine()
    stream = TextToAudioStream(engine)

    # Warmup
    print("Performing system warmup...")
    stream.feed(create_generator("System initialization complete"))
    stream.play(muted=True)

    # Benchmark
    print("Generating audio...")
    gen = create_generator("Hello, my name is Dylan. I'm really cool, and I want to become the best person I can be.")

    start = time()
    stream.feed(gen)
    first_audio_time = None

    def on_audio_frame(_):
        nonlocal first_audio_time
        if first_audio_time is None:
            first_audio_time = time()

    stream.play(log_synthesized_text=True, on_audio_chunk=on_audio_frame)

    if first_audio_time:
        print(f"\nTime to first audio: {first_audio_time - start:.3f} seconds")
    else:
        print("\nNo audio frame callback triggered.")

    print("All generations completed!")


if __name__ == "__main__":
    main()