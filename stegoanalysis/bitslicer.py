import sys
import numpy as np
from PIL import Image

def slice_and_log_bit_differences(img1_path, img2_path, base_name):
    try:
        # 1. Load both images and convert to grayscale
        with Image.open(img1_path).convert('L') as img1, \
             Image.open(img2_path).convert('L') as img2:
            
            if img1.size != img2.size:
                print("Error: Image dimensions must match!")
                return

            arr1 = np.array(img1, dtype=np.uint8)
            arr2 = np.array(img2, dtype=np.uint8)
            total_pixels = arr1.size

        # 2. Calculate the total bitwise difference
        total_xor = np.bitwise_xor(arr1, arr2)

        # Prepare strings for the report
        report_lines = []
        header = f"Bit-plane Difference Report for: {base_name}"
        divider = "-" * 55
        columns = f"{'Bit-plane':<12} | {'Diff Count':<15} | {'Error %':<10}"
        
        report_lines.append(header)
        report_lines.append(divider)
        report_lines.append(columns)
        report_lines.append(divider)

        # 3. Slice the XOR result into 8 bitplanes
        for i in range(8):
            mask = 1 << i
            plane_diff = np.bitwise_and(total_xor, mask)

            # Count the flipped bits
            diff_count = np.count_nonzero(plane_diff)
            error_percentage = (diff_count / total_pixels) * 100

            # Normalize and Invert for PNG (0 -> 255 White, >0 -> 0 Black)
            visual_diff = np.zeros_like(plane_diff)
            visual_diff[plane_diff > 0] = 255
            inverted_plane = (255 - visual_diff).astype(np.uint8)
            
            # Save the PNG
            output_png = f"{base_name}_bit_diff_inv_{i}.png"
            Image.fromarray(inverted_plane).save(output_png)
            
            # Add data to the report
            line = f"Bit {i} (2^{i})    | {diff_count:<15,} | {error_percentage:>8.4f}%"
            report_lines.append(line)

        report_lines.append(divider)
        
        # 4. Save the text report to a file
        report_filename = f"{base_name}_bit_diff_inv.txt"
        with open(report_filename, "w") as f:
            f.write("\n".join(report_lines))

        # 5. Also print to terminal for immediate feedback
        print("\n".join(report_lines))
        print(f"\nResults saved: 8 PNGs and '{report_filename}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python bit_diff_final.py <img1.tif> <img2.tif> <base_name>")
    else:
        slice_and_log_bit_differences(sys.argv[1], sys.argv[2], sys.argv[3])