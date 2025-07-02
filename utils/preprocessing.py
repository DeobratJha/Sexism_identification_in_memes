import torch
from PIL import Image
import easyocr
from utils.models import tokenizer, tokenizer_task2, blip_processor
from transformers import ViTFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
reader = easyocr.Reader(['en'])

def extract_text(image_path):
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

def preprocess_inputs(image_path):
    caption = extract_text(image_path)
    image = Image.open(image_path).convert("RGB")

    image_tensor_task1 = blip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    text_input_task1 = tokenizer([caption], padding=True, truncation=True, return_tensors="pt").to(device)

    image_tensor_task2 = vit_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0).to(device)
    text_input_task2 = tokenizer_task2([caption], padding=True, truncation=True, return_tensors="pt").to(device)

    return image_tensor_task1, text_input_task1, image_tensor_task2, text_input_task2, caption

def run_inference(image_path, task1_model, task2_model):
    img1, txt1, img2, txt2, extracted_caption = preprocess_inputs(image_path)

    with torch.no_grad():
        pred1 = task1_model(img1, txt1, [extracted_caption])
        label1 = torch.argmax(pred1, dim=1).item()

        if label1 == 0:
            return "No (Not Sexist)"
        else:
            pred2 = task2_model(img2, txt2)
            label2 = torch.argmax(pred2, dim=1).item()
            return "Direct" if label2 == 0 else "Judgemental"
