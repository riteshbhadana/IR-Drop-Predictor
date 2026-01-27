import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import UNet
from dataset_loader import get_dataloaders
import matplotlib.pyplot as plt

# ----------------- configs -----------------
BASE_DIR = "./dataset"        # dataset directory relative to src/
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SAMPLE_OUT_DIR = "./samples"
os.makedirs(SAMPLE_OUT_DIR, exist_ok=True)
# -------------------------------------------

def psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10 * torch.log10((data_range ** 2) / mse)

def train():
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE, base_dir=BASE_DIR)
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Use a combo loss: L1 + MSE (regression)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    best_val_mse = 1e9

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for X, Y in train_loader:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            # ensure same shape
            if pred.shape != Y.shape:
                pred = nn.functional.interpolate(pred, size=Y.shape[2:], mode='bilinear', align_corners=False)
            loss = mse(pred, Y) + 0.5 * l1(pred, Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv = Xv.to(DEVICE)
                Yv = Yv.to(DEVICE)
                pred_v = model(Xv)
                if pred_v.shape != Yv.shape:
                    pred_v = nn.functional.interpolate(pred_v, size=Yv.shape[2:], mode='bilinear', align_corners=False)
                l = mse(pred_v, Yv) + 0.5 * l1(pred_v, Yv)
                val_loss += l.item() * Xv.size(0)
                val_mse += mse(pred_v, Yv).item() * Xv.size(0)
                val_psnr += psnr(pred_v, Yv).item() * Xv.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_mse = val_mse / len(val_loader.dataset)
        val_psnr = val_psnr / len(val_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS}  TrainLoss: {epoch_loss:.6f}  ValLoss: {val_loss:.6f}  ValMSE: {val_mse:.6f}  ValPSNR: {val_psnr:.3f}")

        # save sample images for qualitative check (first batch of val)
        if epoch % 5 == 0 or epoch == 1:
            os.makedirs(os.path.join(SAMPLE_OUT_DIR, f"epoch_{epoch}"), exist_ok=True)
            Xs, Ys = next(iter(val_loader))
            Xs = Xs.to(DEVICE)
            Ys = Ys.to(DEVICE)
            pred_sample = model(Xs)
            pred_sample = pred_sample.detach().cpu().numpy()
            Ys = Ys.detach().cpu().numpy()
            Xs = Xs.detach().cpu().numpy()
            # save first 4 samples
            n_save = min(4, Xs.shape[0])
            for i in range(n_save):
                inp = Xs[i]  # 3,x,y
                # combine channels into grid for inspection
                fig, axs = plt.subplots(1,4, figsize=(12,3))
                axs[0].imshow(inp[0], vmin=0, vmax=1); axs[0].set_title("power_grid"); axs[0].axis('off')
                axs[1].imshow(inp[1], vmin=0, vmax=1); axs[1].set_title("cell_density"); axs[1].axis('off')
                axs[2].imshow(inp[2], vmin=0, vmax=1); axs[2].set_title("switching"); axs[2].axis('off')
                axs[3].imshow(pred_sample[i,0], vmin=0, vmax=1); axs[3].set_title("pred_ir_drop"); axs[3].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(SAMPLE_OUT_DIR, f"epoch_{epoch}", f"sample_{i}.png"))
                plt.close(fig)

        # checkpoint saving (by val_mse)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_unet.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_mse": val_mse
            }, ckpt_path)
            print(f"Saved best model -> {ckpt_path}")

    # final save
    final_path = os.path.join(CHECKPOINT_DIR, "final_unet.pth")
    torch.save(model.state_dict(), final_path)
    print("Training finished. Final model saved to:", final_path)

if __name__ == "__main__":
    train()
