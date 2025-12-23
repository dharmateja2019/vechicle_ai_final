import streamlit as st
import cv2
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference.infer_openvino import run_openvino
from inference.infer_pytorch import run_pytorch

st.set_page_config(page_title="Vehicle AI Demo", layout="centered")

st.title("🚗 Vehicle Detection & Color Recognition")
st.write("OpenVINO • PyTorch • Deterministic Color Detection")

# --------------------------------------------------
# Sidebar options
# --------------------------------------------------
st.sidebar.header("Options")

backend = st.sidebar.selectbox(
    "Select Backend",
    ["OpenVINO", "PyTorch", "Compare"]
)

device = st.sidebar.selectbox(
    "OpenVINO Device",
    ["CPU"]
)

uploaded = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Main execution
# --------------------------------------------------
if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.read())

    if st.button("Run Inference"):
        if backend == "OpenVINO":
            vehicle, color, latency = run_openvino("temp.jpg", device)
            img = cv2.imread("output_openvino.jpg")

            st.subheader("Result (OpenVINO)")
            st.write(f"**Vehicle Type:** {vehicle}")
            st.write(f"**Vehicle Color:** {color}")
            st.write(f"**Latency:** {latency:.2f} ms")
            st.image(img, channels="BGR")

        elif backend == "PyTorch":
            vehicle, color, latency = run_pytorch("temp.jpg")
            img = cv2.imread("output_pytorch.jpg")

            st.subheader("Result (PyTorch)")
            st.write(f"**Vehicle Type:** {vehicle}")
            st.write(f"**Vehicle Color:** {color}")
            st.write(f"**Latency:** {latency:.2f} ms")
            st.image(img, channels="BGR")

        elif backend == "Compare":
            st.subheader("Comparison")

            v1, c1, pt_latency = run_pytorch("temp.jpg")
            v2, c2, ov_latency = run_openvino("temp.jpg", device)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### PyTorch")
                st.write(f"Vehicle: {v1}")
                st.write(f"Color: {c1}")
                st.write(f"Latency: {pt_latency:.2f} ms")
                st.image(cv2.imread("output_pytorch.jpg"), channels="BGR")

            with col2:
                st.markdown("### OpenVINO")
                st.write(f"Vehicle: {v2}")
                st.write(f"Color: {c2}")
                st.write(f"Latency: {ov_latency:.2f} ms")
                st.image(cv2.imread("output_openvino.jpg"), channels="BGR")

            st.markdown("---")
            st.write(
                f"**Speedup:** {pt_latency / ov_latency:.2f}x (PyTorch → OpenVINO)"
            )

else:
    st.info("Upload an image to start inference")
