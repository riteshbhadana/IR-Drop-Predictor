from dataset_loader import get_dataloaders

train_loader, val_loader = get_dataloaders()

for batch in train_loader:
    X, Y = batch
    print("Input shape:", X.shape)  # (batch, 3, 64, 64)
    print("Label shape:", Y.shape)  # (batch, 1, 64, 64)
    break
