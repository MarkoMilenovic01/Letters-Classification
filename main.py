# main.py

import os
import glob
import cv2



# --- Configuration ---
BASE_DIR = os.path.join(os.path.dirname(__file__), 'abeceda')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'rough_preprocessed')

# Parameters for bluring the picture and making it white-black
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
THRESH_TYPE     = cv2.THRESH_BINARY_INV
BLOCK_SIZE      = 15
C_CONSTANT      = 8
BLUR_KERNEL = 3

def rough_preprocess(img: cv2.Mat) -> cv2.Mat:
    """
        MEDIAN BLUR APPLIED - Remove scanner dust while perserving the edges
        kernel too large -> you blur the pen strokes
        kernel too small -> noise isn't removed

        ADAPTIVE_THRESH_GAUSSIAN_C - To get the right illumination changes

        THRESH_BINARY_INV - Make letters white, background black

        BLOCK SIZE - SIZE OF LOCAL PATCH

        C - After finding the local avg for brightness, we substract 8 to be sure only the darkest parts stay white
    """
    blurred = cv2.medianBlur(img, BLUR_KERNEL)
    thresh  = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=ADAPTIVE_METHOD,
        thresholdType=THRESH_TYPE,
        blockSize=BLOCK_SIZE,
        C=C_CONSTANT
    )
    return thresh


def process_folder(folder_name: str):

    input_folder  = os.path.join(BASE_DIR, folder_name)
    output_folder = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # only PNG images
    patterns = ['*.png']
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(input_folder, pat)))

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Unable to read: {path}")
            continue
        pre = rough_preprocess(img)
        fname = os.path.basename(path)
        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, pre)
        print(f"[OK]   Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subfolders = [d for d in os.listdir(BASE_DIR)
                  if os.path.isdir(os.path.join(BASE_DIR, d))]
    print(f"Discovered folders: {subfolders}")
    for sub in subfolders:
        print(f"Processing {sub}...")
        process_folder(sub)


if __name__ == '__main__':
    main()
