import sys
from PIL import Image

def jpg_to_grayscale_png(input_path, output_path):
    # Open the input image
    img = Image.open(input_path)
    
    # Convert to true grayscale (single channel, "L" mode in PIL)
    gray_img = img.convert("L")
    
    # Save as PNG
    gray_img.save(output_path, format="PNG")
    print(f"Saved grayscale PNG: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py input.jpg [output.png]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    # Default output filename if not provided
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.rsplit(".", 1)[0] + "_gray.png"
    
    jpg_to_grayscale_png(input_file, output_file)
