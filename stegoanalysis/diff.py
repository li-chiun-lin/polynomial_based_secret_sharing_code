import sys
import numpy as np
from PIL import Image
import pandas as pd

def count_signed_differences(img1_path, img2_path, base_name):
    try:
        # 1. Load and convert to grayscale
        with Image.open(img1_path).convert('L') as img1, \
             Image.open(img2_path).convert('L') as img2:
            
            if img1.size != img2.size:
                print(f"Error: Dimensions mismatch! {img1.size} vs {img2.size}")
                return

            # Convert to NumPy for math 
            # int16 is essential to handle the range -255 to 255
            arr1 = np.array(img1, dtype=np.int16)
            arr2 = np.array(img2, dtype=np.int16)

            # 2. Calculate Signed Difference (A - B)
            # Positive values mean Img1 is brighter; Negative means Img2 is brighter
            diff = arr1 - arr2

            # 3. Calculate counts for every unique difference value
            values, counts = np.unique(diff, return_counts=True)

            # 4. Create a DataFrame for easy viewing and exporting
            df = pd.DataFrame({
                'Difference': values,
                'Count': counts
            })

            # 5. Output to Console
            print(f"\nDifference Analysis for {base_name}:")
            print(f"{'Difference':>12} | {'Pixel Count':>12}")
            print("-" * 27)
            
            # Print all non-zero counts (or you can print the whole table)
            for index, row in df.iterrows():
                print(f"{int(row['Difference']):12d} | {int(row['Count']):12d}")

            # 6. Save results to CSV for further analysis
            output_csv = f"{base_name}_diff_counts.csv"
            df.to_csv(output_csv, index=False)
            print(f"\nFull report saved to: {output_csv}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python diff_count.py <img1.png> <img2.png> <base_name>")
    else:
        count_signed_differences(sys.argv[1], sys.argv[2], sys.argv[3])
