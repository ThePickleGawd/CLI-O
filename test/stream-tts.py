
from RealtimeTTS import TextToAudioStream, KokoroEngine
from time import sleep

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

    # Warmup the engine with a short phrase
    print("Performing system warmup...")
    stream.feed(create_generator("System initialization complete"))
    stream.play(muted=True)

    # Process all test cases automatically
    print("Generating audio...")
    stream.feed(create_generator("Hello, my name is Dylan. I'm really cool, and I want to become the best person I can be. I am learning AI right now. I will work hard and do great things"))
    stream.play(log_synthesized_text=True)

    print("\nAll generations completed!")

if __name__ == "__main__":
    main()