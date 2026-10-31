import cv2
import numpy as np
import argparse
import sys

def generate_inverted_histogram(input_path, output_path):
    # 1. Load the TIFF in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not open or find the image '{input_path}'")
        sys.exit(1)

    # 2. Calculate frequencies (0-255)
    hist_data = np.bincount(img.flatten(), minlength=256)
    
    # 3. Create a WHITE 512x512 canvas (Inverted color)
    # 255 is white in grayscale
    canvas_h, canvas_w = 512, 512
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    # 4. Dynamic Scaling Logic
    max_freq = np.max(hist_data)
    
    if max_freq > 0:
        # 5. Draw the 256 bins (each 2 pixels wide)
        for i in range(256):
            # Calculate height: (count / max_count) * 512
            bin_height = int((hist_data[i] / max_freq) * canvas_h)
            
            if bin_height > 0:
                # Calculate coordinates
                x_start = i * 2
                x_end = x_start + 1 # Ensures 2-pixel width per bin
                
                # Draw black bars (color 0) from bottom up
                # cv2.rectangle(image, pt1, pt2, color, thickness)
                # -1 thickness fills the rectangle
                cv2.rectangle(canvas, 
                             (x_start, canvas_h - 1), 
                             (x_end, canvas_h - bin_height), 
                             0, -1)

    # 6. Save the output
    success = cv2.imwrite(output_path, canvas)
    if success:
        print(f"Histogram saved: {output_path}")
    else:
        print("Error: Could not save the output image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an inverted 512x512 histogram.")
    # Positional arguments (required)
    parser.add_argument("input", help="Path to input TIFF")
    parser.add_argument("output", help="Path to output PNG")
    
    args = parser.parse_args()
    generate_inverted_histogram(args.input, args.output)