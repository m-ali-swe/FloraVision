"""
Streamlit Web Dashboard - Flower Classifier & Image Processing Studio.
A modern, recruiter-ready interface for deep learning computer vision and DIP analysis.
"""

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import io

from src.classifier import FlowerClassifier
from src.image_processor import (
    load_image,
    analyze_image_channels,
    compute_edge_map,
    compute_image_stats,
)
from src.config import CLASS_NAMES, CLASS_COLORS, CLASS_METADATA, TARGET_IMAGE_SIZE

# Page Configuration
st.set_page_config(
    page_title="FloraVision - Deep Learning & DIP Suite",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling for Premium Aesthetics
st.markdown(
    """
    <style>
    /* Dark Theme Customization */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    /* Prediction Cards */
    .pred-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .pred-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.3rem;
        color: white;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    /* Custom Metric Display */
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .metric-lbl {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_classifier():
    """Cache model in memory for fast user response."""
    return FlowerClassifier()


def main():
    # Load model engine
    try:
        classifier = get_classifier()
    except Exception as e:
        st.error(f"Error loading trained Keras model: {e}")
        st.stop()

    # Sidebar Navigation & Settings
    st.sidebar.image(
        "https://img.icons8.com/color/96/000000/flower.png", width=70
    )
    st.sidebar.title("FloraVision DIP")
    st.sidebar.markdown(
        "**Computer Vision & Image Processing Engine**\n"
        "_Powered by TensorFlow 2.x_"
    )
    st.sidebar.divider()

    navigation = st.sidebar.radio(
        "Navigation",
        options=[
            "🌸 Inference Studio",
            "🔬 DIP Analysis Lab",
            "⚡ Model Architecture",
            "ℹ️ Project Information",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("© Digital Image Processing Portfolio")

    # Main Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">FloraVision: Flower Classifier & DIP Suite</div>
            <div class="hero-subtitle">Convolutional Neural Network Inference with Digital Image Processing Diagnostics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # PAGE 1: INFERENCE STUDIO
    if navigation == "🌸 Inference Studio":
        st.subheader("Interactive Flower Classification Studio")
        st.write(
            "Upload an image of a flower to analyze its class and probability distribution using our trained CNN."
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            uploaded_file = st.file_uploader(
                "Choose a flower image...",
                type=["jpg", "jpeg", "png", "webp"],
                help="Supports JPG, JPEG, PNG, WEBP files",
            )

            image = None
            if uploaded_file is not None:
                image = load_image(uploaded_file)
                st.image(
                    image,
                    caption=f"Uploaded Image ({image.size[0]}x{image.size[1]} px)",
                    use_container_width=True,
                )

        with col_right:
            if image is not None:
                st.markdown("### Classification Results")

                with st.spinner("Processing image tensor..."):
                    pred_res = classifier.predict_image(image, top_k=5)

                top_class = pred_res["predicted_class"].title()
                confidence = pred_res["confidence_percentage"]
                bg_color = pred_res["top_k"][0]["color"]

                st.markdown(
                    f"""
                    <div class="pred-card">
                        <div style="color: #94A3B8; font-size: 0.9rem;">TOP PREDICTION</div>
                        <div class="pred-badge" style="background-color: {bg_color};">
                            {top_class} ({confidence}%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Probability Distribution")
                for item in pred_res["top_k"]:
                    name = item["class_name"].title()
                    pct = item["percentage"]
                    col_name, col_bar = st.columns([1, 3])
                    with col_name:
                        st.write(f"**{name}**")
                    with col_bar:
                        st.progress(int(pct), text=f"{pct}%")

                # Scientific Metadata
                meta = pred_res["metadata"]
                if meta:
                    st.divider()
                    st.markdown("#### Botanical Metadata")
                    st.info(
                        f"**Scientific Name:** *{meta.get('scientific_name', 'N/A')}*\n\n"
                        f"**Family:** {meta.get('family', 'N/A')}\n\n"
                        f"**Description:** {meta.get('description', 'N/A')}"
                    )
            else:
                st.info("👆 Please upload an image on the left panel to execute model inference.")

    # PAGE 2: DIP LAB
    elif navigation == "🔬 DIP Analysis Lab":
        st.subheader("Digital Image Processing (DIP) Analysis Lab")
        st.write("Examine image channel distributions, grayscale luminance, and edge maps.")

        dip_file = st.file_uploader(
            "Upload image for DIP analysis", type=["jpg", "jpeg", "png", "webp"], key="dip"
        )

        if dip_file is not None:
            img = load_image(dip_file)
            channels = analyze_image_channels(img)
            edge_map = compute_edge_map(channels["grayscale"])
            stats = compute_image_stats(img)

            # Metric Display Row
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='metric-box'><div class='metric-val'>{stats['dimensions']}</div><div class='metric-lbl'>Dimensions</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-box'><div class='metric-val'>{stats['aspect_ratio']}</div><div class='metric-lbl'>Aspect Ratio</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-box'><div class='metric-val'>{stats['mean_intensity']:.1f}</div><div class='metric-lbl'>Mean Intensity</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-box'><div class='metric-val'>{stats['std_intensity']:.1f}</div><div class='metric-lbl'>Std Dev</div></div>", unsafe_allow_html=True)

            st.divider()
            st.markdown("### Color Channel Decomposition & Gradient Maps")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(channels["red"], caption="Red Channel", use_container_width=True)
            with col2:
                st.image(channels["green"], caption="Green Channel", use_container_width=True)
            with col3:
                st.image(channels["blue"], caption="Blue Channel", use_container_width=True)
            with col4:
                st.image(edge_map, caption="DIP Edge Map", use_container_width=True)
        else:
            st.info("Upload an image above to view digital channel decomposition and spatial gradient analysis.")

    # PAGE 3: MODEL ARCHITECTURE
    elif navigation == "⚡ Model Architecture":
        st.subheader("Model Specifications & Architecture")
        summary_info = classifier.get_model_summary_dict()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Layers", summary_info["total_layers"])
        col2.metric("Input Shape", str(summary_info["input_shape"]))
        col3.metric("Trainable Params", f"{summary_info['trainable_params']:,}")
        col4.metric("Output Classes", summary_info["target_classes"])

        st.divider()
        st.markdown("### Convolutional Neural Network Topology")
        
        layers_data = [
            {"Layer Index": 0, "Layer Name": "sequential_2 (Data Augmentation)", "Output Shape": "(None, 180, 180, 3)", "Type": "RandomFlip/Rotation"},
            {"Layer Index": 1, "Layer Name": "rescaling_3 (Rescaling)", "Output Shape": "(None, 180, 180, 3)", "Type": "1/255 Normalization"},
            {"Layer Index": 2, "Layer Name": "conv2d_6 (Conv2D)", "Output Shape": "(None, 180, 180, 16)", "Type": "Convolution (3x3, 16 filters)"},
            {"Layer Index": 3, "Layer Name": "max_pooling2d_6 (MaxPooling2D)", "Output Shape": "(None, 90, 90, 16)", "Type": "Max Pooling (2x2)"},
            {"Layer Index": 4, "Layer Name": "conv2d_7 (Conv2D)", "Output Shape": "(None, 90, 90, 32)", "Type": "Convolution (3x3, 32 filters)"},
            {"Layer Index": 5, "Layer Name": "max_pooling2d_7 (MaxPooling2D)", "Output Shape": "(None, 45, 45, 32)", "Type": "Max Pooling (2x2)"},
            {"Layer Index": 6, "Layer Name": "conv2d_8 (Conv2D)", "Output Shape": "(None, 45, 45, 64)", "Type": "Convolution (3x3, 64 filters)"},
            {"Layer Index": 7, "Layer Name": "max_pooling2d_8 (MaxPooling2D)", "Output Shape": "(None, 22, 22, 64)", "Type": "Max Pooling (2x2)"},
            {"Layer Index": 8, "Layer Name": "dropout (Dropout)", "Output Shape": "(None, 22, 22, 64)", "Type": "Dropout Regularization"},
            {"Layer Index": 9, "Layer Name": "flatten_2 (Flatten)", "Output Shape": "(None, 30976)", "Type": "Tensor Flattening"},
            {"Layer Index": 10, "Layer Name": "dense_4 (Dense)", "Output Shape": "(None, 128)", "Type": "Fully Connected (ReLU)"},
            {"Layer Index": 11, "Layer Name": "outputs (Dense)", "Output Shape": "(None, 5)", "Type": "Softmax Classification"},
        ]
        st.dataframe(pd.DataFrame(layers_data), use_container_width=True)

    # PAGE 4: ABOUT
    elif navigation == "ℹ️ Project Information":
        st.subheader("About FloraVision Project")
        st.markdown(
            """
            This project demonstrates an end-to-end **Computer Vision & Digital Image Processing** solution built with TensorFlow, Streamlit, and FastAPI.
            
            #### Key Engineering Features:
            - **Custom CNN Architecture**: 3-layer Convolutional Neural Network trained on flower dataset.
            - **Digital Image Processing**: RGB channel analysis, spatial gradient computation, image resizing/normalization.
            - **Multi-Interface Support**: Web Dashboard (Streamlit), REST API (FastAPI), CLI Tool (`cli.py`).
            - **Containerized Deployment**: Docker & Docker Compose setup ready for cloud services.
            - **Automated Testing**: Unit test suite (`pytest`) covering preprocessing, inference, and API endpoints.
            """
        )


if __name__ == "__main__":
    main()