# main.py

import os
import glob
import cv2
import numpy as np




# --- Configuration ---
BASE_DIR = os.path.join(os.path.dirname(__file__), 'abeceda')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'rough_preprocessed')

# Parameters for bluring the picture and making it white-black
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
THRESH_TYPE     = cv2.THRESH_BINARY_INV
BLOCK_SIZE      = 15
C_CONSTANT      = 8
BLUR_KERNEL = 3

# Grid dimensions for GRID detection
TARGET_WIDTH_RATIO = 50
TARGET_HEIGHT_RATIO = 25



#######################  BLURRING ################################
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


#################### FIXING MISSALIGNMENT #####################
# Find the biggest square inside of the paper, assume that's a grid, align it.
def find_sheet_contour(bin_img: np.ndarray) -> np.ndarray:
    '''
        Find the biggest square inside of the paper and assume that's a grid.
    '''
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                max_area = area
                best = approx
    return best.reshape(4, 2) if best is not None else None


def order_points(pts: np.ndarray) -> np.ndarray:
    ''' Get all of the points of the biggest square inside of the paper and order themw'''
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_to_grid(src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    '''
        Make the grid fill the paper by computing the width and height of the new image and applying perspective transform matrix
    '''
    rect = order_points(pts)
    # compute the width of the new image
    (tl, tr, br, bl) = rect
    # horizontal distances
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    # vertical distances
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    # destination points to obtain top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')
    # compute perspective transform matrix and apply
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src, M, (maxWidth, maxHeight))
    return warped


################### MAIN PIPELINE ######################3
def process_folder(folder: str):
    '''Detect sheet, deskew and threshold each PNG, then save aligned binary.'''
    inp = os.path.join(BASE_DIR, folder)
    out = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(out, exist_ok=True)

    for path in glob.glob(os.path.join(inp, '*.png')):
        img = cv2.imread(path)
        if img is None:
            print(f'[WARN] Cannot read {path}')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_initial = rough_preprocess(gray)
        sheet = find_sheet_contour(bin_initial)
        if sheet is None:
            aligned = bin_initial.copy()
            print(f'[WARN] No sheet border found for {path}, skipping warp')
        else:
            warped = warp_to_grid(gray, sheet)
            aligned = rough_preprocess(warped)
        fname = os.path.basename(path)
        cv2.imwrite(os.path.join(out, fname), aligned)
        print(f'[OK] {folder}/{fname}: aligned and saved')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    print('Discovered folders:', subs)
    for s in subs:
        process_folder(s)


if __name__ == '__main__':
    main()
