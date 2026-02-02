import torch
import whisper
from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import io

# ================= CONFIG =================
BASE_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
ADAPTER_PATH = "/teamspace/studios/this_studio/smolvlm_News_flood_finetuned"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

whisper_model = whisper.load_model("base")

app = FastAPI()

# ================= IMAGE Q&A =================
@app.post("/image_qa")
async def image_qa(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    if not question.strip():
        question = "What is happening in this image?"

    prompt = f"<image> {question}"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return {"answer": answer}

# ================= AUDIO Q&A =================
@app.post("/audio_qa")
async def audio_qa(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    audio_bytes = await file.read()
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    transcript = whisper_model.transcribe("temp_audio.wav")["text"]

    if not question.strip():
        question = "What disaster-related information is mentioned?"

    prompt = f"""
    Audio Transcript:
    {transcript}

    Question:
    {question}

    Answer:
    """

    inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    answer = processor.batch_decode(output, skip_special_tokens=True)[0]

    return {
        "transcript": transcript,
        "answer": answer
    }

