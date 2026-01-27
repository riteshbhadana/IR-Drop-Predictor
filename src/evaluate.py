import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet
from dataset_loader import get_dataloaders
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "./checkpoints/best_unet.pth"
OUT_DIR = "./eval_samples"
os.makedirs(OUT_DIR, exist_ok=True)

def load_model(path):
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    ck = torch.load(path, map_location=DEVICE)
    if "model_state" in ck:
        model.load_state_dict(ck["model_state"])
    else:
        model.load_state_dict(ck)
    model.eval()
    return model

def psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * torch.log10((data_range ** 2) / mse)

def evaluate():
    _, val_loader = get_dataloaders(batch_size=8, base_dir="./dataset")
    model = load_model(CHECKPOINT)
    mse_loss = nn.MSELoss()
    tot_mse = 0.0
    tot_psnr = 0.0
    n = 0
    with torch.no_grad():
        for X, Y in val_loader:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            pred = model(X)
            if pred.shape != Y.shape:
                pred = nn.functional.interpolate(pred, size=Y.shape[2:], mode='bilinear', align_corners=False)
            b = X.shape[0]
            tot_mse += mse_loss(pred, Y).item() * b
            tot_psnr += psnr(pred, Y).item() * b
            # save first batch visualizations
            pred_np = pred.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()
            X_np = X.detach().cpu().numpy()
            for i in range(min(4, b)):
                fig, axs = plt.subplots(1,4, figsize=(12,3))
                axs[0].imshow(X_np[i,0], vmin=0, vmax=1); axs[0].set_title("power_grid"); axs[0].axis('off')
                axs[1].imshow(X_np[i,1], vmin=0, vmax=1); axs[1].set_title("cell_density"); axs[1].axis('off')
                axs[2].imshow(Y_np[i,0], vmin=0, vmax=1); axs[2].set_title("true_ir_drop"); axs[2].axis('off')
                axs[3].imshow(pred_np[i,0], vmin=0, vmax=1); axs[3].set_title("pred_ir_drop"); axs[3].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f"val_sample_{n}_{i}.png"))
                plt.close(fig)
            n += b
    avg_mse = tot_mse / n
    avg_psnr = tot_psnr / n
    print(f"Validation MSE: {avg_mse:.6f}  Avg PSNR: {avg_psnr:.3f}")

if __name__ == "__main__":
    evaluate()
