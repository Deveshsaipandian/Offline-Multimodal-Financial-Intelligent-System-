import gradio as gr
import torch
import whisper
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

# ======================================================
# MODEL CONFIG
# ======================================================
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

# Whisper
whisper_model = whisper.load_model("base")

# ======================================================
# AUDIO → TRANSCRIPT → Q&A
# ======================================================
def audio_qa(audio_file, question):
    if audio_file is None:
        return "❌ Please upload an audio file.", ""

    # 1. Transcribe
    result = whisper_model.transcribe(audio_file.name)
    transcript = result["text"]

    if not question or question.strip() == "":
        question = "What disaster-related information is mentioned?"

    # 2. Ask model using transcript
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

    return transcript, answer.strip()

# ======================================================
# IMAGE Q&A
# ======================================================
def image_qa(image, question):
    if image is None:
        return "❌ Upload an image."

    if not question or question.strip() == "":
        question = "What is happening in this image?"

    prompt = f"<image> {question}"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    return processor.decode(output[0], skip_special_tokens=True)

# ======================================================
# UI
# ======================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Multimodal Disaster Intelligence System  
    **Image | Audio | Text (RAG)**  
    This UI confirms deployment is working.
    """)

    with gr.Tabs():

        # ================= IMAGE TAB =================
        with gr.Tab("🖼️ Image"):
            img = gr.Image(type="pil", label="Upload Image")
            img_q = gr.Textbox(
                label="Ask a question about the image",
                placeholder="e.g. Is there flooding here?"
            )
            img_out = gr.Textbox(label="Answer")

            gr.Button("Analyze Image").click(
                image_qa, inputs=[img, img_q], outputs=img_out
            )

        # ================= AUDIO TAB =================
        with gr.Tab("🎧 Audio"):

            audio_file = gr.File(
                label="Upload Audio / Video (mp3, wav, mp4, m4a)",
                file_types=["audio", "video"]
            )

            audio_question = gr.Textbox(
                label="Ask a question about the audio",
                placeholder="e.g. What disaster is mentioned?"
            )

            transcript = gr.Textbox(label="Transcript")
            audio_answer = gr.Textbox(label="Answer")

            gr.Button("Analyze Audio").click(
                audio_qa,
                inputs=[audio_file, audio_question],
                outputs=[transcript, audio_answer]
            )

        # ================= TEXT TAB =================
        with gr.Tab("📄 Text / RAG"):
            txt = gr.Textbox(lines=6, label="Enter text")
            txt_out = gr.Textbox(label="Output")

            gr.Button("Analyze Text").click(lambda x: x, txt, txt_out)

# ======================================================
# LAUNCH
# ======================================================
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)
