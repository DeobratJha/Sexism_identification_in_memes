# Sexism_identification_in_memes
# 🧠 Meme Classifier
# my research paper related to this [Sexism Identification](https://ceur-ws.org/Vol-4038/paper_157.pdf)

A deep learning-based meme classification app that automatically detects sexist content in memes using both image and text features. The project uses:
- **EasyOCR** to extract meme text from images
- **BLIP** + **BERT** for initial classification (`yes` / `no`)
- **ViT** + **English BERT** for fine classification (`direct` / `judgemental`)
- **Gradio** for an interactive web interface

---

## 🧰 Features

- 🖼️ Upload meme images directly
- 🔍 Extracts text automatically using EasyOCR
- 🤖 Classifies memes using a two-stage model pipeline:
  - Task 1: Detect if meme is sexist (`yes` / `no`)
  - Task 2: If `yes`, further classifies it as `direct` or `judgemental`
- 🧪 Simple Gradio interface for real-time inference

---



## 🖼️ Demo UI Preview

![Gradio UI Screenshot](ui_preview.png)



