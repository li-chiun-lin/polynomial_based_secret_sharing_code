import sys
import numpy as np
from PIL import Image

def create_xor_pngs(img1_path, img2_path, base_name):
    try:
        # 1. Load and convert to grayscale (L)
        with Image.open(img1_path).convert('L') as img1, \
             Image.open(img2_path).convert('L') as img2:
            
            if img1.size != img2.size:
                print(f"Error: Dimensions mismatch! {img1.size} vs {img2.size}")
                return

            # Convert to NumPy arrays
            arr1 = np.array(img1, dtype=np.uint8)
            arr2 = np.array(img2, dtype=np.uint8)

            # 2. Calculate Bitwise XOR Difference: A ^ B
            xor_result = np.bitwise_xor(arr1, arr2)

            # --- Generate the 4 variations ---

            # 1. Pure XOR Difference
            diff_pure = xor_result

            # 2. Inverted XOR (255 - XOR)
            diff_inv = (255 - xor_result)

            # 3. Scaled XOR (XOR * 20, clamped at 255)
            # Use int16 to prevent overflow before the clip
            diff_20 = np.clip(xor_result.astype(np.int16) * 20, 0, 255).astype(np.uint8)

            # 4. Inverted Scaled XOR (255 - (XOR * 20))
            diff_inv_20 = (255 - diff_20).astype(np.uint8)

            # --- Save all 4 files as PNG ---
            files_to_save = {
                f"{base_name}_xor_diff.png": diff_pure,
                f"{base_name}_xor_diff_inv.png": diff_inv,
                f"{base_name}_xor_diff20.png": diff_20,
                f"{base_name}_xor_diff_inv20.png": diff_inv_20
            }

            for filename, data in files_to_save.items():
                Image.fromarray(data).save(filename, "PNG")
                print(f"Saved: {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python xor_to_png.py <img1.tiff> <img2.tiff> <base_name>")
    else:
        # User provides 2 tiff inputs and a string for the base name
        create_xor_pngs(sys.argv[1], sys.argv[2], sys.argv[3])