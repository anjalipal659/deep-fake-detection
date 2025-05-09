# 🧠 Deepfake Detection using Vision Transformer (ViT)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Live%20App-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/AnjaliPal06/deep-fake-detection)

This is a **Deepfake Detection web application** that uses a fine-tuned Vision Transformer model (`google/vit-base-patch16-224`) from Hugging Face to determine whether a facial image is **Real** or **Fake** (deepfake). Built with **PyTorch**, **Gradio**, and deployed on **Hugging Face Spaces**, the app provides instant predictions with visual confidence levels.

---

## 🌐 Live Demo

👉 **Try the App Here**: [https://huggingface.co/spaces/AnjaliPal06/deep-fake-detection](https://huggingface.co/spaces/AnjaliPal06/deep-fake-detection)

---

## 📱 About the App

This app allows users to upload face images and checks whether the image is a **real photo** or a **deepfake**. It uses `google/vit-base-patch16-224`, a transformer-based image model, fine-tuned for binary classification. The app displays prediction results along with real/fake **confidence scores** in an intuitive, responsive Gradio interface.

---

## ✅ Features

- 🖼️ Upload face images directly via drag-and-drop
- 🔍 Predicts whether the image is Real or Fake
- 📊 Shows confidence levels using progress bars
- 🧠 Uses fine-tuned ViT transformer model for classification
- 🌐 Deployed on Hugging Face Spaces
- 🖥️ Also supports local execution

---

## 🧪 Tech Stack & Modules Used

| Module | Purpose |
|--------|---------|
| **Transformers** (`google/vit-base-patch16-224`) | Vision Transformer from Hugging Face for image embeddings |
| **PyTorch** | For model fine-tuning and inference |
| **Gradio** | To build a lightweight, interactive web UI |
| **OpenCV (`cv2`)** | For image resizing and preprocessing |
| **NumPy** | For array and matrix operations |
| **Torchvision** | For image transformations and preprocessing |
| **CNN layers** | Additional convolutional layers for better classification |

---

## 📂 Project Structure

```text
deep-fake-detection/
├── app.py                # Main executable file – launches the Gradio app
├── main.py               # Model logic – loads the ViT model, predicts Real/Fake
├── requirements.txt      # List of dependencies to run the app
├── .gitattributes        # Required for Hugging Face Spaces repository setup
├── best-hf-model/        # Folder containing fine-tuned model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   ├── tokenizer_config.json
│   └── ... (other Hugging Face model files)
```
