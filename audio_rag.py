import whisper
import os

model = whisper.load_model("small")  # fully offline after download

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

