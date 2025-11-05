import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import gdown  # to download from Google Drive

# --- Page setup ---
st.set_page_config(page_title="Steel Surface Defect Detector", page_icon="🧱")

st.title("Steel Surface Defect Detection System")
st.write("Upload an image of a steel surface to detect potential defects.")

# --- Google Drive model link ---
model_url = "https://drive.google.com/uc?export=download&id=1Ov9tpdU7q8PP6fTKrzbwiaRov0tdRtAh"
model_path = "model_best.pth"

# --- Download model if not present ---
if not os.path.exists(model_path):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(model_url, model_path, quiet=False)

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: {predicted.item()}")
