# ⚡ IR-Drop Prediction — Summary Report

## 1. Project Overview
This project presents a **deep learning–based approach** for predicting **IR-drop** in integrated circuits using a **CNN-based U-Net architecture**.  
Traditional IR-drop signoff tools (e.g., Cadence Voltus) provide accurate analysis but are computationally expensive and typically used late in the design flow.  

The objective of this project is to develop a **fast, AI-driven surrogate model** that enables **early-stage IR-drop estimation**, helping designers identify potential risk regions before running full signoff analysis.

---

## 2. Dataset Summary
Each design sample is represented using four NumPy (`.npy`) feature maps, each of size **64×64**:

| File Name | Description | Shape |
|----------|-------------|-------|
| `input_power_grid.npy` | Power grid strength / effective resistance distribution | 64×64 |
| `input_cell_density.npy` | Standard cell placement density (current demand proxy) | 64×64 |
| `input_switching.npy` | Switching activity representing dynamic current | 64×64 |
| `labels_ir_drop.npy` | Ground-truth IR-drop heatmap | 64×64 |

All maps are normalized to the range **[0, 1]** and used as **pixel-wise spatial features**.

---

## 3. IR-Drop Theory (Simplified)
IR-drop refers to the reduction in supply voltage caused by current flowing through resistive elements of the power delivery network.

### Ohm’s Law
\[
IR\_drop = I \times R
\]

### Industry-Level Solver Equation
Commercial EDA tools solve numerical formulations such as:
\[
\nabla \cdot (\sigma \nabla V) = -J
\]

Where:
- **V** = voltage  
- **σ** = metal conductivity  
- **J** = current density  

Solving these equations over large power grids is accurate but computationally intensive.

---

## 4. Synthetic Ground Truth Generation
Since real signoff IR-drop data is not available during early design stages, ground-truth labels are generated using a **physics-inspired approximation**:

\[
IR\_drop = \frac{Switching \times CellDensity}{PowerGrid}
\]

This is followed by:
- Gaussian smoothing to model spatial voltage diffusion  
- Normalization for stable training  

This approach produces **realistic IR-drop patterns**, including localized hotspots.

---

## 5. Model Architecture (U-Net)
The IR-drop prediction task is formulated as a **pixel-wise regression problem** and solved using a **U-Net architecture**, consisting of:

- Encoder for hierarchical feature extraction  
- Bottleneck layer for global context  
- Decoder for spatial reconstruction  
- Skip connections to preserve fine-grained layout details  

U-Net is well-suited for dense spatial prediction tasks such as IR-drop heatmap generation.

---

## 6. Training Configuration

| Parameter | Value |
|---------|-------|
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Epochs | 30 |
| Batch Size | 16 |

The model is trained end-to-end using backpropagation and gradient descent.

---

## 7. Evaluation Results

| Metric | Value |
|------|------|
| **Validation MSE** | ≈ 0.00049 |
| **PSNR** | ≈ 33.36 dB |
| **Hotspot Detection** | Accurate |

The model achieves strong numerical and spatial accuracy, successfully identifying voltage droop regions and IR-drop hotspots.

---

## 8. Streamlit Application
A Streamlit-based application is developed to demonstrate practical usability. The interface supports:

- Uploading `.npy` input maps  
- Auto-loading dataset samples  
- Visualization of input features  
- Predicted IR-drop heatmaps  
- Optional comparison with ground truth  
- Automatic summary generation  

The application is designed for **demo readiness and rapid experimentation**.

---

## 9. Conclusion
This project demonstrates the feasibility of using **deep learning as a surrogate model** for early-stage IR-drop estimation.  
By reframing IR-drop analysis as a spatial regression problem, the proposed approach enables **fast pre-signoff screening**, complementing traditional signoff tools and reducing design iteration time.

---

## 10. Author
**Ritesh**
