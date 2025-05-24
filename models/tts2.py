
from RealtimeTTS import TextToAudioStream, KokoroEngine

# Structured data containing voices and their corresponding texts
TEST_CASES = [
    {
        "voice": "zoe",  # Female
        "text": "Don't you just hate it when <laugh> your cat wakes you up like this? Meow. <laugh> Meow. Meow. <chuckle> Meow."
    },
    {
        "voice": "tara",  # Female
        "text": "Asked my assistant to stop talking. Now it's just <laugh> whispering: \"null, null, null...\""
    },
    {
        "voice": "mia",  # Female
        "text": "Told my assistant I need dating pickup lines. It said <laugh> \"Are you a router? Because I'm connecting.\""
    },
    {
        "voice": "jess",  # Male
        "text": "I told my assistant a horror story. It <laugh> got so scared it <chuckle> switched to Comic Sans."
    }
]

def create_generator(text):
    """Create a text generator for a single text string"""
    def generator():
        yield text
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
    for case in TEST_CASES:        
        print("Generating audio...")
        stream.feed(create_generator(case["text"]))
        stream.play(log_synthesized_text=True)

    print("\nAll generations completed!")

if __name__ == "__main__":
    main()