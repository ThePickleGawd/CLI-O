from RealtimeSTT import AudioToTextRecorder
import sounddevice as sd

sd.default.device = (1, None)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # your existing code here
    recorder = AudioToTextRecorder(
        enable_realtime_transcription=True,
        silero_use_onnx=False,
        no_log_file=True,
    )

    print("Say something...")
    print("Transcribed:", recorder.text())