#@title 3. Load Model from HF Directory and Launch Gradio Interface

# --- Imports ---
import torch
import gradio as gr
from PIL import Image
import os
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, ViTForImageClassification # Still need ViT class
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch import device, cuda
import numpy as np

# --- Configuration ---
# REMOVED: pkl_file_path = '/content/finetune_vit_model.pkl'
# ADDED: Path to the directory created by unzipping
hf_model_directory = 'best-model-hf'
# IMPORTANT: This MUST match the base model used for fine-tuning to load the correct feature extractor
model_checkpoint = "google/vit-base-patch16-224"
device_to_use = device('cuda' if cuda.is_available() else 'cpu')
print(f"Using device: {device_to_use}")

# --- Global variables for loaded components ---
inference_model = None
inference_feature_extractor = None
inference_transforms = None
inference_id2label = None
num_labels = 0

# --- Load Feature Extractor (Needed for Preprocessing) ---
# (This part remains the same as it's needed for transforms regardless of model loading method)
try:
    print(f"Loading feature extractor for: {model_checkpoint}")
    inference_feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    print("Feature extractor loaded successfully.")

    # --- Define Image Transforms (Must match inference transforms from training) ---
    normalize = Normalize(mean=inference_feature_extractor.image_mean, std=inference_feature_extractor.image_std)
    if isinstance(inference_feature_extractor.size, dict):
       image_size = inference_feature_extractor.size.get('shortest_edge', inference_feature_extractor.size.get('height', 224))
    else:
       image_size = inference_feature_extractor.size
    print(f"Using image size: {image_size}")

    inference_transforms = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        normalize,
    ])
    print("Inference transforms defined.")

except Exception as e:
    print(f"Error loading feature extractor or defining transforms: {e}")
    print("Cannot proceed without feature extractor and transforms.")
    raise SystemExit("Feature extractor/transforms loading failed.")


# --- Load the Fine-Tuned Model from Hugging Face Save Directory --- ## MODIFIED BLOCK ##
if not os.path.isdir(hf_model_directory):
    print(f"ERROR: Hugging Face model directory not found at '{hf_model_directory}'.")
    print("Please ensure you uploaded the zip file and ran the 'Unzip' cell successfully.")
    inference_model = None # Ensure model is None if dir not found
else:
    print(f"Attempting to load model from directory: {hf_model_directory}")
    try:
        # Load the model using from_pretrained with the directory path
        inference_model = ViTForImageClassification.from_pretrained(hf_model_directory)

        # --- Post-Load Setup ---
        inference_model.to(device_to_use)
        inference_model.eval() # Set model to evaluation mode
        print("Model loaded successfully from directory and moved to device.")

        # Try to get label mapping from the loaded model's config (this usually works well with from_pretrained)
        if hasattr(inference_model, 'config') and hasattr(inference_model.config, 'id2label'):
            inference_id2label = inference_model.config.id2label
            # Ensure keys are integers if loaded from JSON/dict
            inference_id2label = {int(k): v for k, v in inference_id2label.items()}
            num_labels = len(inference_id2label)
            print(f"Loaded id2label mapping from model config: {inference_id2label}")
            print(f"Number of labels: {num_labels}")
        else:
            # Fallback if id2label isn't in the config for some reason
            print("WARNING: Could not find 'id2label' in the loaded model's config.")
            # --- !! MANUALLY DEFINE LABELS HERE IF NEEDED !! ---
            # Example: Replace with your actual labels and order
            inference_id2label = {0: 'fake', 1: 'real'} # Make sure this matches your training
            num_labels = len(inference_id2label)
            print(f"Using manually defined id2label: {inference_id2label}")
            # -----------------------------------------------------

        if num_labels == 0:
            print("ERROR: Number of labels is zero. Cannot proceed.")
            inference_model = None # Prevent Gradio launch

    except Exception as e:
        # Catch errors during from_pretrained (e.g., missing files, config errors)
        print(f"An unexpected error occurred loading the model from directory: {e}")
        inference_model = None # Ensure model is None on error
## --- END OF MODIFIED BLOCK --- ##


# --- Define the Prediction Function for Gradio ---
# (This function remains the same)
def predict(image: Image.Image):
    """
    Takes a PIL image, preprocesses it, and returns label probabilities.
    """
    # Ensure model and necessary components are loaded
    if inference_model is None:
        return {"Error": "Model not loaded. Please check loading logs."}
    if inference_transforms is None:
        return {"Error": "Inference transforms not defined."}
    if inference_id2label is None:
         return {"Error": "Label mapping (id2label) not available."}
    if image is None:
        return None # Gradio handles None input gracefully sometimes

    try:
        # Preprocess the image
        image = image.convert("RGB") # Ensure 3 channels
        pixel_values = inference_transforms(image).unsqueeze(0).to(device_to_use)

        # Perform inference
        with torch.no_grad():
            outputs = inference_model(pixel_values=pixel_values)
            logits = outputs.logits

        # Get probabilities and format output
        probabilities = F.softmax(logits, dim=-1)[0] # Get probabilities for the first (only) image
        confidences = {inference_id2label[i]: float(prob) for i, prob in enumerate(probabilities)}
        return confidences

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Return error in a format Gradio Label can display
        return {"Error": f"Prediction failed: {str(e)}"}


# --- Create and Launch the Gradio Interface ---
# (This part remains the same, but title/description updated slightly)
if inference_model and inference_id2label and num_labels > 0:
    print("\nSetting up Gradio Interface...")
    try:
        iface = gr.Interface(
            fn=predict,
            inputs=gr.Image(type="pil", label="Upload Face Image"),
            outputs=gr.Label(num_top_classes=num_labels, label="Prediction (Real/Fake)"),
            # Updated Title/Description
            title="Real vs. Fake Face Detector)",
            description="Upload an image of a face to classify it as real or fake using the fine-tuned ViT model loaded from the 'best-model-hf' directory.",
            # examples=[...] # Optional: Add example image paths if you upload some
        )

        print("Launching Gradio interface...")
        print("Access the interface through the public URL generated below (if sharing is enabled) or the local URL.")
        iface.launch(share=True, debug=True, show_error=True)

    except Exception as e:
        print(f"Error creating or launching Gradio interface: {e}")

else:
    print("\nCould not launch Gradio interface because the model or label mapping failed to load.")
    print("Please check the error messages above.")

# Keep the cell running to keep the Gradio interface active
print("\nGradio setup finished. Interface should be running or an error reported above.")
print("Stop this cell execution in Colab to shut down the Gradio server.")