import streamlit as st
import torch
from pathlib import Path
from PIL import Image

from dataset import get_transforms
from model import build_model


# Page settings
st.set_page_config(
    page_title="Road Scene Object Classification",
    page_icon="🚗",
    layout="centered"
)


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load trained model
@st.cache_resource
def load_model():

    checkpoint_path = Path("outputs") / "best_model.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint["classes"]

    model, _ = build_model(
        num_classes=len(classes),
        freeze_backbone=False
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, classes


model, classes = load_model()


# Interface
st.title("🚗 Road Scene Object Classification")

st.write(
    "Upload a road-scene image and let the AI model classify the object."
)

st.divider()


uploaded_file = st.file_uploader(
    "📁 Upload a road-scene image",
    type=["jpg", "jpeg", "png"]
)


# Prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Classify Image"):

        _, eval_tf = get_transforms()

        x = eval_tf(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            probabilities = torch.softmax(
                model(x),
                dim=1
            )[0].cpu()

        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_class = classes[predicted_index]

        confidence = (
            probabilities[predicted_index].item()
            * 100
        )

        st.divider()

        st.subheader("🎯 Prediction")

        st.success(
            f"Predicted Class: {predicted_class.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("📊 Class Probabilities")

        for i, class_name in enumerate(classes):

            probability = (
                probabilities[i].item() * 100
            )

            st.write(
                f"{class_name}: {probability:.2f}%"
            )

            st.progress(
                int(probability)
            )
