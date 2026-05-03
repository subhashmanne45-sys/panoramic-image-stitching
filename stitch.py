"""
stitch.py — CLI entry point for Panoramic Image Stitching
----------------------------------------------------------
Usage:
    python stitch.py

You will be prompted to enter the number of images and their file paths
in left-to-right order. The stitched panorama and keypoint match
visualization are saved to the output/ folder.

Alternatively, pass image paths directly as arguments:
    python stitch.py inputs/img1.jpg inputs/img2.jpg inputs/img3.jpg
"""

import sys
import os
import cv2
import imutils
from panorama import Panaroma


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_and_validate_image(path):
    """Load an image from disk and raise a clear error if it fails."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: '{path}'")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image (unsupported format?): '{path}'")
    return image


def resize_image(image, target_width=400, target_height=400):
    """
    Resize an image to a fixed width then height so all images share
    the same aspect-normalized dimensions before stitching.
    """
    image = imutils.resize(image, width=target_width)
    image = imutils.resize(image, height=target_height)
    return image


def ensure_output_dir(path="output"):
    """Create the output directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def collect_image_paths_from_args():
    """Return image paths passed as command-line arguments (sys.argv[1:])."""
    return sys.argv[1:]


def collect_image_paths_interactively():
    """Prompt the user to enter image paths one by one."""
    try:
        n = int(input("Enter the number of images you want to concatenate: ").strip())
    except ValueError:
        print("[ERROR] Please enter a valid integer.")
        sys.exit(1)

    if n < 2:
        print("[ERROR] You need at least 2 images to create a panorama.")
        sys.exit(1)

    print("Enter the image names with extension in order of left to right:")
    paths = []
    for i in range(n):
        path = input(f"  Enter image {i + 1} path (with extension): ").strip()
        paths.append(path)

    return paths


# ------------------------------------------------------------------
# Core stitching logic
# ------------------------------------------------------------------

def stitch_all(images):
    """
    Stitch a list of images (left to right) into a single panorama.

    Stitches right-to-left pair-by-pair, accumulating the result.

    Returns:
        (panorama_image, matched_points_image)
    """
    panorama = Panaroma()
    n = len(images)

    print(f"\n[INFO] Starting stitching of {n} images...")

    if n == 2:
        print("  → Stitching image 1 + image 2 ...")
        output = panorama.image_stitch([images[0], images[1]], match_status=True)
    else:
        # Stitch the last two first, then progressively prepend the rest
        print(f"  → Stitching image {n - 1} + image {n} ...")
        output = panorama.image_stitch([images[n - 2], images[n - 1]], match_status=True)

        if output is None:
            raise RuntimeError("Stitching failed at the first pair. Images may not overlap enough.")

        result, matched_points = output

        for i in range(n - 2):
            idx = n - i - 3
            print(f"  → Stitching image {idx + 1} into the panorama ...")
            output = panorama.image_stitch([images[idx], result], match_status=True)
            if output is None:
                raise RuntimeError(
                    f"Stitching failed when adding image {idx + 1}. "
                    "Check that images overlap sufficiently and are in the correct order."
                )
            result, matched_points = output

    if output is None:
        raise RuntimeError("Stitching failed. Ensure images have enough overlapping regions.")

    return output  # (result, matched_points)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # 1. Collect image paths
    if len(sys.argv) > 1:
        paths = collect_image_paths_from_args()
        print(f"[INFO] Using {len(paths)} image(s) from command-line arguments.")
    else:
        paths = collect_image_paths_interactively()

    if len(paths) < 2:
        print("[ERROR] At least 2 images are required.")
        sys.exit(1)

    # 2. Load and validate images
    print("\n[INFO] Loading images...")
    images = []
    for path in paths:
        try:
            img = load_and_validate_image(path)
            img = resize_image(img)
            images.append(img)
            print(f"  ✓  {path}  →  shape after resize: {img.shape}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    # 3. Stitch
    try:
        result, matched_points = stitch_all(images)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # 4. Save outputs
    ensure_output_dir("output")
    panorama_path = os.path.join("output", "panorama_image.jpg")
    matches_path = os.path.join("output", "matched_points.jpg")

    cv2.imwrite(panorama_path, result)
    cv2.imwrite(matches_path, matched_points)

    print(f"\n[SUCCESS] Panorama saved to:       {panorama_path}")
    print(f"[SUCCESS] Match visualization at:  {matches_path}")

    # 5. Display (only if a display is available)
    try:
        cv2.imshow("Panorama", result)
        cv2.imshow("Keypoint Matches", matched_points)
        print("\nPress any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("[INFO] No display available — images saved to output/ folder only.")


if __name__ == "__main__":
    main()
