#@title 3. Load Model from HF Directory and Launch Gradio Interface

# --- Imports ---
import torch
import gradio as gr
from PIL import Image
import os
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, ViTForImageClassification
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch import device, cuda
import numpy as np

# --- Configuration ---
hf_model_directory = 'best-model-hf'  # Corrected path (no leading dot)
model_checkpoint = "google/vit-base-patch16-224"
device_to_use = device('cuda' if cuda.is_available() else 'cpu')
print(f"Using device: {device_to_use}")

# --- Predictor Class ---
class ImagePredictor:
    def __init__(self, model_dir, base_checkpoint, device):
        self.model_dir = model_dir
        self.base_checkpoint = base_checkpoint
        self.device = device
        self.model = None
        self.feature_extractor = None
        self.transforms = None
        self.id2label = None
        self.num_labels = 0
        self._load_resources() # Load everything during initialization

    def _load_resources(self):
        print("--- Loading Predictor Resources ---")
        # --- Load Feature Extractor (Needed for Preprocessing) ---
        try:
            print(f"Loading feature extractor for: {self.base_checkpoint}")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.base_checkpoint)
            print("Feature extractor loaded.")

            # --- Define Image Transforms ---
            normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
            if isinstance(self.feature_extractor.size, dict):
               image_size = self.feature_extractor.size.get('shortest_edge', self.feature_extractor.size.get('height', 224))
            else:
               image_size = self.feature_extractor.size
            print(f"Using image size: {image_size}")

            self.transforms = Compose([
                Resize(image_size),
                CenterCrop(image_size),
                ToTensor(),
                normalize,
            ])
            print("Inference transforms defined.")

        except Exception as e:
            print(f"FATAL: Error loading feature extractor or defining transforms: {e}")
            # Re-raise to prevent using a partially initialized object
            raise RuntimeError("Feature extractor/transforms loading failed.") from e

        # --- Load the Fine-Tuned Model ---
        if not os.path.isdir(self.model_dir):
            print(f"FATAL: Model directory not found at '{self.model_dir}'.")
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        print(f"Attempting to load model from directory: {self.model_dir}")
        try:
            self.model = ViTForImageClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            print("Model loaded successfully from directory and moved to device.")

            # --- Load Label Mapping ---
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
                self.num_labels = len(self.id2label)
                print(f"Loaded id2label mapping from model config: {self.id2label}")
                print(f"Number of labels: {self.num_labels}")
            else:
                print("WARNING: Could not find 'id2label' in the loaded model's config.")
                # --- !! MANUALLY DEFINE FALLBACK IF NEEDED !! ---
                self.id2label = {0: 'fake', 1: 'real'} # ENSURE THIS MATCHES TRAINING
                self.num_labels = len(self.id2label)
                print(f"Using manually defined id2label: {self.id2label}")
                # ----------------------------------------------

            if self.num_labels == 0:
                raise ValueError("Number of labels is zero after loading.")

            print("--- Predictor Resources Loaded Successfully ---")

        except Exception as e:
            print(f"FATAL: An unexpected error occurred loading the model: {e}")
            # Reset model attribute to indicate failure clearly
            self.model = None
            # Re-raise to prevent using a partially initialized object
            raise RuntimeError("Model loading failed.") from e

    # --- Prediction Method ---
# Inside the ImagePredictor class:
def predict(self, image: Image.Image):
    print("--- Predict function called ---") # Check if this even prints in Space logs
    if image is None:
        print("Input image is None")
        return None
    try:
        # Simulate some processing time
        import time
        time.sleep(0.1)
        # Return a dummy dictionary, bypassing all model/transform logic
        dummy_output = {"fake": 0.6, "real": 0.4} # Use your actual labels
        print(f"Returning dummy output: {dummy_output}")
        return dummy_output
    except Exception as e:
        print(f"Error in *simplified* predict: {e}")
        return {"Error": f"Simplified prediction failed: {str(e)}"}


# --- Main Execution Logic ---
predictor = None
try:
    # Instantiate the predictor ONCE globally
    # This loads the model, tokenizer, transforms, etc. immediately
    predictor = ImagePredictor(
        model_dir=hf_model_directory,
        base_checkpoint=model_checkpoint,
        device=device_to_use
    )
except Exception as e:
     print(f"Failed to initialize ImagePredictor: {e}")
     # predictor remains None


# --- Create and Launch the Gradio Interface ---
if predictor and predictor.model: # Check if predictor initialized successfully
    print("\nSetting up Gradio Interface...")
    try:
        iface = gr.Interface(
            # Pass the INSTANCE METHOD to fn
            fn=predictor.predict,
            inputs=gr.Image(type="pil", label="Upload Face Image"),
            outputs=gr.Label(num_top_classes=predictor.num_labels, label="Prediction (Real/Fake)"),
            title="Real vs. Fake Face Detector",
            description=f"Upload an image of a face to classify it using the fine-tuned ViT model loaded from the '{hf_model_directory}' directory.",
        )

        print("Launching Gradio interface...")
        # Set share=True as requested
        iface.launch(share=True, debug=True, show_error=True).queue()

    except Exception as e:
        print(f"Error creating or launching Gradio interface: {e}")

else:
    print("\nCould not launch Gradio interface because the Predictor failed to initialize.")
    print("Please check the error messages above.")


# Optional: Add message for Colab/persistent running if needed
print("\nGradio setup finished. Interface should be running or an error reported above.")
# print("Stop this cell execution in Colab to shut down the Gradio server.")