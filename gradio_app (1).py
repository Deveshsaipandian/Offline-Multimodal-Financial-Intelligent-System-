import gradio as gr
import torch
from PIL import Image
import whisper

from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

# -----------------------------
# MODEL LOADING
# -----------------------------
BASE_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
ADAPTER_PATH = "/teamspace/studios/this_studio/smolvlm_News_flood_finetuned"

processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Whisper for audio
whisper_model = whisper.load_model("base")

# -----------------------------
# IMAGE FUNCTION (VLM)
# -----------------------------
def image_fn(image, question):
    if image is None:
        return "❌ Please upload an image."

    if question is None or question.strip() == "":
        question = "What is happening in this image?"

    prompt = f"<image> {question}"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    answer = processor.batch_decode(output, skip_special_tokens=True)[0]
    return answer.strip()

# -----------------------------
# AUDIO FUNCTION
# -----------------------------
def audio_fn(audio_path):
    if audio_path is None:
        return "❌ Please upload an audio file."

    result = whisper_model.transcribe(audio_path)
    return result["text"]

# -----------------------------
# TEXT / RAG FUNCTION (SIMPLE)
# -----------------------------
def text_fn(text):
    if text is None or text.strip() == "":
        return "❌ Please enter some text."

    return f"Processed text:\n{text}"

# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌊 Multimodal Disaster Intelligence System  
        **Image | Audio | Text (RAG)**  
        This UI confirms deployment is working.
        """
    )

    with gr.Tabs():

        # -------- IMAGE TAB --------
        with gr.Tab("🖼️ Image"):
            img = gr.Image(type="pil", label="Upload Disaster Image")
            img_question = gr.Textbox(
                label="Ask a question about the image",
                placeholder="e.g. Is there flooding? How severe is it?"
            )
            img_out = gr.Textbox(label="Model Answer")
            gr.Button("Analyze Image").click(
                image_fn,
                inputs=[img, img_question],
                outputs=img_out
            )

        # -------- AUDIO TAB --------
        with gr.Tab("🎧 Audio"):
            aud = gr.Audio(type="filepath", label="Upload Audio")
            aud_out = gr.Textbox(label="Transcript")
            gr.Button("Analyze Audio").click(audio_fn, aud, aud_out)


        # -------- TEXT / RAG TAB --------
        with gr.Tab("📄 Text / RAG"):
            txt = gr.Textbox(lines=6, label="Enter text or document content")
            txt_out = gr.Textbox(label="Output")
            gr.Button("Analyze Text").click(text_fn, txt, txt_out)

# -----------------------------
# LAUNCH
# -----------------------------
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)
