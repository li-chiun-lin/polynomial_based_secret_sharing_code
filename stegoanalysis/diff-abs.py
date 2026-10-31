import sys
import numpy as np
from PIL import Image

def create_abs_diff_pngs(img1_path, img2_path, base_name):
    try:
        # 1. Load and convert to grayscale
        with Image.open(img1_path).convert('L') as img1, \
             Image.open(img2_path).convert('L') as img2:
            
            if img1.size != img2.size:
                print(f"Error: Dimensions mismatch! {img1.size} vs {img2.size}")
                return

            # Convert to NumPy for math (int16 avoids negative number errors)
            arr1 = np.array(img1, dtype=np.int16)
            arr2 = np.array(img2, dtype=np.int16)

            # 2. Calculate Numerical Absolute Difference: |A - B|
            abs_diff = np.abs(arr1 - arr2)

            # --- Generate the 4 variations ---

            # 1. Pure Difference
            diff_pure = abs_diff.astype(np.uint8)

            # 2. Inverted Difference (255 - Pure)
            diff_inv = (255 - abs_diff).astype(np.uint8)

            # 3. Scaled Difference (Pure * 20, clamped at 255)
            diff_20 = np.clip(abs_diff * 20, 0, 255).astype(np.uint8)

            # 4. Inverted Scaled Difference (255 - (Pure * 20))
            diff_inv_20 = (255 - np.clip(abs_diff * 20, 0, 255)).astype(np.uint8)

            # --- Save all 4 files as PNG ---
            files_to_save = {
                f"{base_name}_abs_diff.png": diff_pure,
                f"{base_name}_abs_diff_inv.png": diff_inv,
                f"{base_name}_abs_diff20.png": diff_20,
                f"{base_name}_abs_diff_inv20.png": diff_inv_20
            }

            for filename, data in files_to_save.items():
                Image.fromarray(data).save(filename, "PNG")
                print(f"Saved: {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python abs_to_png.py <img1.tiff> <img2.tiff> <base_name>")
    else:
        create_abs_diff_pngs(sys.argv[1], sys.argv[2], sys.argv[3])