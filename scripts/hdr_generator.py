import numpy as np
import argparse
import sys
import os

import cv2  # OpenCV for robust HDR write support (.hdr)

def generate_hdr_image(width: int, height: int) -> np.ndarray:
    """
    Generates a NumPy array with random, visually plausible HDR data.

    Returns a float32 array (H, W, 3) in RGB order with values that can exceed 1.0.
    """
    print("Generating image data...")

    block_size = 16
    small_w = (width + block_size - 1) // block_size
    small_h = (height + block_size - 1) // block_size

    # Base image in a moderate range
    small_image = np.random.uniform(low=0.1, high=0.7, size=(small_h, small_w, 3))
    base_image = np.kron(small_image, np.ones((block_size, block_size, 1)))
    final_image = base_image[:height, :width, :].copy()

    # Bright HDR speckles
    num_speckles = max(1, (width * height) // 500)
    y_coords = np.random.randint(0, height, size=num_speckles)
    x_coords = np.random.randint(0, width, size=num_speckles)

    intensity = np.random.uniform(20.0, 100.0, size=(num_speckles, 1))
    speckle_colors = intensity * np.array([[1.0, 1.0, 0.9]], dtype=np.float32)
    final_image[y_coords, x_coords] = speckle_colors

    return final_image.astype(np.float32)  # RGB, float32


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random, visible HDR image for compression testing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("width", type=int, help="Width of the output image in pixels.")
    parser.add_argument("height", type=int, help="Height of the output image in pixels.")
    parser.add_argument("output_file", type=str, help="Path to save the output .hdr file (e.g., 'test_image.hdr').")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        print("Error: Width and height must be positive integers.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output_file
    if not output_path.lower().endswith(".hdr"):
        print(
            "Warning: Output filename does not end with .hdr. "
            "Saving as Radiance HDR by appending '.hdr' to the name.",
            file=sys.stderr
        )
        output_path = args.output_file + ".hdr"

    # Ensure the directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        print(f"Error: Directory does not exist: {out_dir}", file=sys.stderr)
        sys.exit(1)

    hdr_rgb = generate_hdr_image(args.width, args.height)  # float32, RGB

    # OpenCV uses BGR channel order
    hdr_bgr = hdr_rgb[..., ::-1].copy()  # float32, BGR

    try:
        print(f"Saving image to {output_path}...")
        ok = cv2.imwrite(output_path, hdr_bgr)
        if not ok:
            print("Error: OpenCV failed to write the .hdr file.", file=sys.stderr)
            sys.exit(1)
        print("✅ Done!")
    except Exception as e:
        print(f"Error: Could not save the file. {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()