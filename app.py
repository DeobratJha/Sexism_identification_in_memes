import gradio as gr
import os
from utils.models import load_task1_model, load_task2_model
from utils.preprocessing import run_inference, extract_text

import os, gdown

os.makedirs("checkpoints", exist_ok=True)

# Task 1
if not os.path.exists("checkpoints/best_model_task1_rank(1).pt"):
    gdown.download("https://drive.google.com/file/d/19YWvJQ-ussNtRHqWRSE6gmlkRBXeFE5o/view?usp=sharing", quiet=False)

# Task 2
if not os.path.exists("checkpoints/best_model_checkpoint_task2.pth"):
    gdown.download("https://drive.google.com/file/d/1ZVa6cXH9XUOYgDQ0E5APMdxIw9KQbUpp/view?usp=sharing", "checkpoints/best_model_checkpoint_task2.pth", quiet=False)


task1_model = load_task1_model()
task2_model = load_task2_model()

def classify_meme(image):
    # Save uploaded image
    temp_path = "static/uploads/temp_image.jpg"
    image.save(temp_path)

    # Extract text and run both models
    extracted_text = extract_text(temp_path)
    result = run_inference(temp_path, task1_model, task2_model)
    return result, extracted_text

demo = gr.Interface(
    fn=classify_meme,
    inputs=gr.Image(type="pil", label="Upload Meme Image"),
    outputs=[
        gr.Textbox(label="Classification Result"),
        gr.Textbox(label="Extracted Text")
    ],
    title="Meme Classifier",
    description="Upload a meme to classify whether it's sexist. Auto extracts text using EasyOCR and classifies using two models.",
)

if __name__ == "__main__":
    demo.launch()
