import sys
import os
import numpy as np
from PIL import Image

def slice_bitplanes(input_path, base_name):
    try:
        # 1. Load the image and ensure it is grayscale
        with Image.open(input_path).convert('L') as img:
            arr = np.array(img, dtype=np.uint8)

        # 2. Iterate through each bit (0 to 7)
        for i in range(8):
            # Bitwise AND to isolate the i-th bit
            # (1 << i) creates a "mask": bit 0 is 1, bit 1 is 2, bit 2 is 4, etc.
            plane = np.bitwise_and(arr, 1 << i)

            # 3. Convert to binary visual (0 or 255)
            # If the bit was present, it will be > 0. We set it to 255 (White).
            plane[plane > 0] = 255
            
            # 4. Save the result
            output_name = f"{base_name}_bitplane_{i}.png"
            Image.fromarray(plane).save(output_name)
            print(f"Saved: {output_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bitplanes.py <input.tif> <output_base_name>")
    else:
        slice_bitplanes(sys.argv[1], sys.argv[2])
