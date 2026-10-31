import sys
from PIL import Image

def convert_tiff_to_png(input_tiff, output_png):
    try:
        # Open the TIFF file
        with Image.open(input_tiff) as img:
            # Convert to RGB if it's a specialized TIFF format (like CMYK or YCbCr)
            # If it's already grayscale or RGB, this ensures PNG compatibility
            rgb_img = img.convert("RGB")
            
            # Save as PNG
            rgb_img.save(output_png, "PNG")
            print(f"Success: {input_tiff} converted to {output_png}")
            
    except Exception as e:
        print(f"Error: Could not convert file. {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input.tif> <output.png>")
    else:
        convert_tiff_to_png(sys.argv[1], sys.argv[2])