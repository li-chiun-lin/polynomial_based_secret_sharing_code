import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- RS Analysis Functions ---

def fluctuation(g):
    """Calculates the noisiness (fluctuation) of a pixel group."""
    return np.sum(np.abs(g[1:] - g[:-1]))

def F1(x):
    """LSB flipping function: 0<->1, 2<->3, 4<->5..."""
    return np.where(x % 2 == 0, x + 1, x - 1)

def Fm1(x):
    """Dual flipping function: -1<->0, 1<->2, 3<->4..."""
    return np.where(x % 2 == 1, x + 1, x - 1)

def rs_calculate(img, mask):
    """Categorizes pixel groups into Regular (Rm) and Singular (Sm)."""
    img = img.astype(np.int32)
    n = len(mask)
    pixels = img.flatten()
    num_groups = len(pixels) // n
    groups = pixels[:num_groups * n].reshape((num_groups, n))
    
    f_orig = np.array([fluctuation(g) for g in groups])
    
    flipped_groups = groups.copy()
    for i in range(n):
        if mask[i] == 1:
            flipped_groups[:, i] = F1(flipped_groups[:, i])
        elif mask[i] == -1:
            flipped_groups[:, i] = Fm1(flipped_groups[:, i])
    
    f_flipped = np.array([fluctuation(g) for g in flipped_groups])
    
    rm = np.sum(f_flipped > f_orig) / num_groups
    sm = np.sum(f_flipped < f_orig) / num_groups
    return rm, sm

# --- Experiment and File Output ---

def run_rs_experiment(original_path, stego_path, base_name):
    # Load images
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    if orig is None or stego is None:
        print(f"Error: Could not load images. Check paths: {original_path}, {stego_path}")
        return

    rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = [] # To store data for the .txt file
    rm_list, sm_list, r_m_list, s_m_list = [], [], [], []
    
    mask = np.array([0, 1, 1, 0])
    neg_mask = np.array([0, -1, -1, 0])
    
    header = f"{'Rate (%)':<10} | {'Rm':<10} | {'Sm':<10} | {'R-m':<10} | {'S-m':<10}"
    print(f"\nProcessing: {base_name}")
    print(header)
    print("-" * 60)

    for r in rates:
        h, w = orig.shape
        sampling_mask = np.random.rand(h, w) < r
        img_r = orig.copy()
        img_r[sampling_mask] = stego[sampling_mask]
        
        rm, sm = rs_calculate(img_r, mask)
        r_m, s_m = rs_calculate(img_r, neg_mask)
        
        rm_list.append(rm)
        sm_list.append(sm)
        r_m_list.append(r_m)
        s_m_list.append(s_m)
        
        row_str = f"{r*100:>7.0f}%    | {rm:.6f} | {sm:.6f} | {r_m:.6f} | {s_m:.6f}"
        print(row_str)
        results.append(row_str)

    # 1. Save Data to .txt
    txt_filename = f"{base_name}.txt"
    with open(txt_filename, "w") as f:
        f.write(f"RS Analysis Data for: {base_name}\n")
        f.write(f"Original: {original_path}\nStego: {stego_path}\n\n")
        f.write(header + "\n")
        f.write("-" * 60 + "\n")
        for line in results:
            f.write(line + "\n")
    print(f"\nData saved to: {txt_filename}")

    # 2. Save Plot to .png
    
    plt.figure(figsize=(10, 7))
    plt.plot(rates, rm_list, 'r-s', label='$R_M$')
    plt.plot(rates, sm_list, 'b-s', label='$S_M$')
    plt.plot(rates, r_m_list, 'r--o', label='$R_{-M}$')
    plt.plot(rates, s_m_list, 'b--o', label='$S_{-M}$')
    
    plt.title(f'RS Steganalysis Curves: {base_name}', fontsize=14)
    plt.xlabel('Embedding Rate (p)', fontsize=12)
    plt.ylabel('Percentage of Groups', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    png_filename = f"{base_name}.png"
    plt.savefig(png_filename, dpi=300)
    print(f"Plot saved to: {png_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("\nUsage: python rs_script.py <original_img> <stego_img> <base_name>")
        print("Example: python rs_script.py cover.png stego.png experiment_1")
    else:
        run_rs_experiment(sys.argv[1], sys.argv[2], sys.argv[3])