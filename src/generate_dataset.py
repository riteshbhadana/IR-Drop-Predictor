import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Create folders
os.makedirs("dataset/input_power_grid", exist_ok=True)
os.makedirs("dataset/input_cell_density", exist_ok=True)
os.makedirs("dataset/input_switching", exist_ok=True)
os.makedirs("dataset/labels_ir_drop", exist_ok=True)

IMG_SIZE = 64
NUM_SAMPLES = 1000

def generate_gaussian_map(size, n_clusters=3):
    """Generate clustered Gaussian blob maps."""
    img = np.zeros((size, size))
    for _ in range(n_clusters):
        x, y = np.random.randint(0, size, 2)
        sigma = np.random.randint(5, 15)
        xv, yv = np.meshgrid(np.arange(size), np.arange(size))
        blob = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
        img += blob
    img = gaussian_filter(img, sigma=2)
    return (img - img.min()) / (img.max() - img.min())

def generate_perlin_noise(size):
    """Generate smooth activity patterns."""
    noise = np.random.normal(0, 1, (size, size))
    noise = gaussian_filter(noise, sigma=3)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

print("Generating 1000 synthetic IR-drop samples...")

for i in range(NUM_SAMPLES):
    # Input 1: Power grid density (stronger grid = lower IR drop)
    power_grid = 1 - generate_gaussian_map(IMG_SIZE, n_clusters=5)

    # Input 2: Cell density map
    cell_density = generate_gaussian_map(IMG_SIZE, n_clusters=5)

    # Input 3: Switching activity
    switching = generate_perlin_noise(IMG_SIZE)

    # Output label: IR drop = (switching * cell_density) / power grid
    ir_drop = (switching * cell_density) / (power_grid + 0.1)
    ir_drop = gaussian_filter(ir_drop, sigma=3)
    ir_drop = (ir_drop - ir_drop.min()) / (ir_drop.max() - ir_drop.min())

    # Save maps
    np.save(f"dataset/input_power_grid/{i}.npy", power_grid)
    np.save(f"dataset/input_cell_density/{i}.npy", cell_density)
    np.save(f"dataset/input_switching/{i}.npy", switching)
    np.save(f"dataset/labels_ir_drop/{i}.npy", ir_drop)

print("Dataset created successfully!")
