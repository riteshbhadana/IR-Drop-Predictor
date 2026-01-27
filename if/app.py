import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os, sys

# ==================================================
# PATH SETUP (ROBUST & PROFESSIONAL)
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from model import UNet

# ==================================================
# CONFIGURATION
# ==================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = os.path.join(SRC_DIR, "checkpoints", "best_unet.pth")
DATASET_DIR = os.path.join(SRC_DIR, "dataset")
SAMPLE_LIMIT = 2000

st.set_page_config(
    page_title="AI IR-Drop Predictor",
    layout="wide"
)

# ==================================================
# SESSION STATE
# ==================================================
if "gt" not in st.session_state:
    st.session_state.gt = None

if "mode" not in st.session_state:
    st.session_state.mode = "manual"

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

model = load_model()

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
    <h1 style="text-align:center;">⚡ AI IR-Drop Predictor</h1>
    <p style="text-align:center; color:gray;">
    Deep Learning–Based Early IR-Drop Estimation
    </p>
    """,
    unsafe_allow_html=True
)

st.write(
    "This application performs **deep learning inference** using a trained "
    "CNN-based U-Net model to predict spatial IR-drop heatmaps."
)

# ==================================================
# SIDEBAR — DATASET SAMPLES
# ==================================================
st.sidebar.header("Dataset Samples")
use_sample = st.sidebar.checkbox("Use dataset samples", value=True)

sample_index = None
if use_sample and os.path.isdir(DATASET_DIR):
    pg_dir = os.path.join(DATASET_DIR, "input_power_grid")
    files = sorted([
        f.replace(".npy", "")
        for f in os.listdir(pg_dir)
        if f.endswith(".npy")
    ])
    files = files[:SAMPLE_LIMIT]

    if files:
        sample_index = st.sidebar.selectbox("Select sample index", files)

# ==================================================
# FILE UPLOAD
# ==================================================
col1, col2, col3 = st.columns(3)
with col1:
    uploaded_pg = st.file_uploader("Upload power_grid.npy", type=["npy"])
with col2:
    uploaded_cd = st.file_uploader("Upload cell_density.npy", type=["npy"])
with col3:
    uploaded_sw = st.file_uploader("Upload switching.npy", type=["npy"])

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def load_dataset_sample(idx):
    pg = np.load(os.path.join(DATASET_DIR, "input_power_grid", f"{idx}.npy"))
    cd = np.load(os.path.join(DATASET_DIR, "input_cell_density", f"{idx}.npy"))
    sw = np.load(os.path.join(DATASET_DIR, "input_switching", f"{idx}.npy"))
    label = np.load(os.path.join(DATASET_DIR, "labels_ir_drop", f"{idx}.npy"))
    return pg, cd, sw, label


def resize_prediction(pred, target_shape):
    pred_t = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
    pred_t = F.interpolate(
        pred_t,
        size=target_shape,
        mode="bilinear",
        align_corners=False
    )
    return pred_t.squeeze().cpu().numpy()

# ==================================================
# PREDICTION
# ==================================================
if st.button("Predict IR-Drop"):

    # ----------------------------------------------
    # INPUT MODE
    # ----------------------------------------------
    is_dataset = use_sample and sample_index is not None

    if is_dataset:
        pg, cd, sw, gt = load_dataset_sample(sample_index)
        st.session_state.gt = gt
        st.session_state.mode = "dataset"
    else:
        if not (uploaded_pg and uploaded_cd and uploaded_sw):
            st.error("Upload all three input files or select a dataset sample.")
            st.stop()

        pg = np.load(uploaded_pg)
        cd = np.load(uploaded_cd)
        sw = np.load(uploaded_sw)

        st.session_state.gt = None
        st.session_state.mode = "manual"

    if not (pg.shape == cd.shape == sw.shape):
        st.error("All input maps must have the same shape.")
        st.stop()

    # ----------------------------------------------
    # MODEL INFERENCE (FORWARD PASS)
    # ----------------------------------------------
    X = np.stack([pg, cd, sw]).astype(np.float32)
    X_t = torch.tensor(X).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(X_t).squeeze().cpu().numpy()

    gt = st.session_state.gt

    if gt is not None and pred.shape != gt.shape:
        pred = resize_prediction(pred, gt.shape)

    # ==================================================
    # INPUT VISUALIZATION
    # ==================================================
    st.subheader("Input Maps")
    cols = st.columns(3)
    titles = ["Power Grid", "Cell Density", "Switching Activity"]

    for col, data, title in zip(cols, [pg, cd, sw], titles):
        with col:
            plt.figure(figsize=(3, 3))
            plt.imshow(data, cmap="inferno")
            plt.axis("off")
            st.caption(title)
            st.pyplot(plt)

    # ==================================================
    # OUTPUT VISUALIZATION
    # ==================================================
    st.subheader("Prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("Ground Truth IR-Drop")
        if st.session_state.mode == "dataset" and gt is not None:
            plt.imshow(gt, cmap="inferno")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("Ground truth unavailable in inference mode")

    with c2:
        st.caption("Predicted IR-Drop")
        plt.imshow(pred, cmap="inferno")
        plt.axis("off")
        st.pyplot(plt)

    with c3:
        st.caption("Difference Map (GT − Pred)")
        if st.session_state.mode == "dataset" and gt is not None:
            diff = gt - pred
            plt.imshow(diff, cmap="bwr")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("Difference requires ground truth.")

    # ==================================================
    # SUMMARY
    # ==================================================
    st.subheader("📊 Inference Summary (Deep Learning Output)")

    max_drop = float(pred.max())
    avg_drop = float(pred.mean())
    hotspots = int(np.sum(pred > 0.7))  # conservative early-stage threshold

    if max_drop < 0.3:
        risk = "LOW RISK"
    elif max_drop < 0.6:
        risk = "MODERATE RISK"
    else:
        risk = "HIGH RISK"

    st.markdown(f"""
- **Max IR-Drop:** {max_drop:.4f}  
- **Avg IR-Drop:** {avg_drop:.4f}  
- **Hotspots (>0.7):** {hotspots}  
- **Risk Level:** **{risk}**
""")

    st.success("Inference completed successfully.")
