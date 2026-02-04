<h1 align="center">⚡ IR-Drop Prediction Using Deep Learning</h1>

<p align="center">
  <b>CNN-based U-Net Surrogate Model for Early-Stage IR-Drop Estimation</b><br/>
  <i>Deep Learning • VLSI • EDA • Power Integrity</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit"/>
  <img src="https://img.shields.io/badge/Model-U--Net-blue"/>
  <img src="https://img.shields.io/badge/Task-Pixel--wise%20Regression-success"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen"/>
</p>

---

## 📌 Project Overview

IR-drop is a critical **power integrity challenge** in modern VLSI designs, where excessive voltage drop can lead to timing failures and functional issues.  
While commercial signoff tools such as Cadence Voltus and Ansys RedHawk provide accurate analysis, they rely on computationally expensive numerical solvers and are typically used late in the design flow.

This project explores a **deep learning–based surrogate modeling approach** for **early-stage IR-drop estimation**.  
The problem is reformulated as a **pixel-wise spatial regression task**, where a CNN-based **U-Net** learns a direct mapping from layout-level features to dense IR-drop heatmaps.

The goal is **not to replace signoff tools**, but to provide a **fast pre-signoff screening mechanism** that enables rapid design iteration.

---

## 🧮 IR-Drop Theory (High-Level)

At a basic level, IR-drop follows Ohm's law:

```
IR_drop = I × R
```

In industry-grade tools, IR-drop analysis is performed by solving partial differential equations over the power grid:

```
∇ · (σ ∇V) = −J
```

Where:
- **V** represents the voltage distribution
- **σ** is metal conductivity
- **J** is current density

Solving these equations using numerical methods is accurate but time-consuming, motivating the use of learned surrogate models.

---

## 📂 Project Structure

```text
ir_drop_project/
│
├── src/
│   ├── model.py
│   ├── dataset_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── checkpoints/
│       └── best_unet.pth
│
├── ui/
│   └── app.py
│
├── dataset/
│   ├── input_power_grid/
│   ├── input_cell_density/
│   ├── input_switching/
│   └── labels_ir_drop/
│
├── reports/
│   └── IR_Drop_Full_Report.pdf
│
└── README.md
```

---

## 📦 Dataset Description

Each design sample is represented using four NumPy (.npy) files, each of size 64×64, forming a structured spatial representation of the chip layout:

| File | Description | Data Type | Shape |
|------|-------------|-----------|-------|
| input_power_grid.npy | Power grid strength / effective resistance | Float | 64×64 |
| input_cell_density.npy | Standard cell placement density (current demand proxy) | Float | 64×64 |
| input_switching.npy | Switching activity representing dynamic current | Float | 64×64 |
| labels_ir_drop.npy | Ground-truth IR-drop heatmap | Float | 64×64 |

All maps are normalized to the range [0, 1] and treated as pixel-wise input features.

---

## 🧪 Synthetic Ground Truth Generation

Ground-truth IR-drop labels are generated using a physics-inspired approximation:

```
IR_drop = (Switching × CellDensity) / PowerGrid
```

The result is smoothed using a Gaussian filter and normalized to [0, 1], producing realistic IR-drop patterns with spatial voltage gradients and hotspots.

---

## 🧬 Model Architecture (CNN-based U-Net)

The IR-drop prediction task is formulated as image-to-image regression and solved using a U-Net architecture:

```
Input  (3 × 64 × 64)
   ↓
Encoder Blocks (spatial feature extraction)
   ↓
Bottleneck (global context)
   ↓
Decoder Blocks (spatial reconstruction)
   ↓
Output (1 × 64 × 64 IR-drop heatmap)
```

Skip connections preserve fine-grained spatial information, which is critical for accurate hotspot localization.

---

## 🧠 Key Deep Learning Features

✔ Pixel-wise IR-drop regression  
✔ End-to-end CNN training (no handcrafted rules)  
✔ Single forward-pass inference  
✔ Dense heatmap prediction  
✔ Spatial hotspot detection  
✔ Surrogate modeling of physics-based analysis  

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 30 |
| Batch Size | 16 |

---

## 📊 Experimental Results

| Metric | Value |
|--------|-------|
| Validation MSE | ≈ 0.00049 |
| PSNR | ≈ 33.36 dB |
| Hotspot Detection | Accurate |
| Inference Time | Milliseconds |

The model demonstrates strong numerical accuracy and spatial consistency while enabling real-time inference.

---

## 🖥️ Streamlit Application

A Streamlit-based interface is developed to demonstrate practical usability:

- Upload .npy input maps or auto-load dataset samples
- Visualize power grid, density, and switching maps
- View predicted IR-drop heatmaps
- Compare predictions with ground truth (when available)
- Inspect difference maps and summary statistics

**Run the application:**
```bash
cd if
streamlit run app.py
```

---

## 📄 Reports Included

✔ Full Technical Project Report (PDF)  
✔ Summary Report  

---

## 🧑‍💻 Author

**Ritesh**  
BTech — Artificial Intelligence

