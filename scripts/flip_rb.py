import argparse
import sys
from PIL import Image

def flip_rb_channels(input_path: str, output_path: str) -> None:
    """
    Opens an image, flips its Red and Blue channels, and saves the result.

    Args:
        input_path (str): The path to the source image file.
        output_path (str): The path where the modified image will be saved.
    """
    try:
        # Open the input image
        with Image.open(input_path) as img:
            print(f"Processing '{input_path}'...")

            # Ensure the image is in a mode that has R, G, B channels (e.g., RGB, RGBA)
            if img.mode not in ('RGB', 'RGBA'):
                print(f"Error: Image mode '{img.mode}' is not supported for channel swapping.")
                print("Only RGB and RGBA images can be processed. Aborting.")
                sys.exit(1)

            # Split the image into its individual channels
            # For RGB: (R, G, B)
            # For RGBA: (R, G, B, A)
            channels = list(img.split())

            # Swap the Red and Blue channels
            # The Red channel is at index 0, and the Blue channel is at index 2
            channels[0], channels[2] = channels[2], channels[0]

            # Merge the swapped channels back into a new image
            swapped_img = Image.merge(img.mode, channels)

            # Save the new image to the specified output path
            swapped_img.save(output_path)
            print(f"✅ Success! Image saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="A script to flip the Red (R) and Blue (B) channels of an image.",
        epilog="Example: python swap_channels.py input.jpg output.png"
    )
    parser.add_argument("input_image", help="The path to the input image file.")
    parser.add_argument("output_image", help="The path for the output image file. The extension determines the format.")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the main function with the provided arguments
    flip_rb_channels(args.input_image, args.output_image)