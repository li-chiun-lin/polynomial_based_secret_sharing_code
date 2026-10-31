import sys
import numpy as np
from PIL import Image

def generate_diff_list(img1_path, img2_path, output_csv):
    try:
        # 1. Load images and convert to grayscale
        with Image.open(img1_path).convert('L') as img1, \
             Image.open(img2_path).convert('L') as img2:
            
            if img1.size != img2.size:
                print("Error: Dimensions mismatch!")
                return

            # 2. Calculate signed difference: (A - B)
            # Use int16 to accommodate the range [-255, 255]
            arr1 = np.array(img1, dtype=np.int16)
            arr2 = np.array(img2, dtype=np.int16)
            diff = arr1 - arr2 

        # 3. Count frequencies for every value from -255 to 255
        # bincount only works on non-negative numbers, so we use np.unique
        # or a simple loop for clarity and 100% accuracy on the range.
        values, counts = np.unique(diff, return_counts=True)
        
        # Create a dictionary for quick lookup
        count_dict = dict(zip(values, counts))

        # 4. Write to CSV
        with open(output_csv, 'w') as f:
            # Header
            f.write("Difference,Count\n")
            
            # Iterate through the full range to ensure all 511 pairs are present
            for d in range(-255, 256):
                count = count_dict.get(d, 0) # Get count or 0 if that diff didn't occur
                f.write(f"{d},{count}\n")

        print(f"Success! List of 511 pairs saved to {output_csv}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python diff_list.py <img1.tif> <img2.tif> <output.csv>")
    else:
        generate_diff_list(sys.argv[1], sys.argv[2], sys.argv[3])
