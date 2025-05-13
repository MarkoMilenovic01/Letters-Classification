# main.py

import os
import glob
import cv2
import numpy as np
import pandas as pd

# --- Configuration ---
BASE_DIR = os.path.join(os.path.dirname(__file__), 'abeceda')
PREP_DIR = os.path.join(os.path.dirname(__file__), 'rough_preprocessed')
CELL_ROOT  = os.path.join(os.path.dirname(__file__), 'cells')


# PARAMETERS FOR PRE-PREPROCESSING, BLURING THE PICTURE, MAKING IT BLACK WHITE ETC...
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
THRESH_TYPE     = cv2.THRESH_BINARY_INV
BLOCK_SIZE      = 15
C_CONSTANT      = 8
BLUR_KERNEL = 3

# GRID
NUM_ROWS = 50
NUM_COLS = 25

# CELL WIDTH AND CELL HEIGHT => FILE DIMENSIONS
CELL_W    = 50
CELL_H    = 50
TARGET_W  = NUM_COLS * CELL_W
TARGET_H  = NUM_ROWS * CELL_H


# SLOVENIAN ALPHABET - Cannot use C S and Z because of the map
SLOVENIAN = ['A','B','C','C-sumik','D','E','F','G','H','I','J','K','L','M','N','O','P','R','S','S-sumik','T','U','V','Z','Z-sumik']
TYPE_MAP = {
    'AA':  'printed_big',
    'aSm': 'printed_small',
    'plg': 'handwritten_big',
    'psm': 'handwritten_small',
}

ALPHABET = (SLOVENIAN * ((NUM_COLS // len(SLOVENIAN)) + 1))[:NUM_COLS]



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
    ''' Get all of the points of the biggest square inside of the paper and order them'''
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_to_grid(src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
     GET THE BIGGEST SQUARE, ORDER THE POINTS, COMPUTE EUCLIDEAN DISTANCE AND GET THE SKEW MATRIX TO TRANSFORM THE PAGE.
    """
    rect = order_points(pts)

    dst = np.array([
        [0,         0        ],
        [TARGET_W-1, 0        ],
        [TARGET_W-1, TARGET_H-1],
        [0,         TARGET_H-1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src, M, (TARGET_W, TARGET_H))
    return warped

################### MAIN PIPELINE FOR FOLDER ######################
def process_folder(folder: str):
    inp  = os.path.join(BASE_DIR, folder)
    outp = os.path.join(PREP_DIR, folder)
    os.makedirs(outp, exist_ok=True)

    for path in glob.glob(os.path.join(inp, '*.png')):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] can't read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin0  = rough_preprocess(gray)
        sheet = find_sheet_contour(bin0)
        if sheet is None:
            aligned = bin0
        else:
            warped  = warp_to_grid(gray, sheet)
            aligned = rough_preprocess(warped)
        fname = os.path.basename(path)
        cv2.imwrite(os.path.join(outp, fname), aligned)
        print(f"[OK] {folder}/{fname} → aligned")


################ SEGMENT LETTERS #############3
MARGIN = 5

def correct_rotation(img: np.ndarray, orient_threshold: float = 1.5) -> np.ndarray:
    """
        IF THERE IS MUCH MORE HORIZONTAL ENERGY, THEN ROTATE IT 90 DEGREES
        ALL LETTERS HAVE PREDOMINANTLY UPRIGHT ENERGY
    """
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    mag_x = np.sum(np.abs(gx))
    mag_y = np.sum(np.abs(gy))

    if mag_y > orient_threshold * mag_x:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def is_empty_cell(img: np.ndarray) -> bool:
    """ IS THERE EMPTY CELL, WITHOUT WHITE PIXELS"""
    return cv2.countNonZero(img) == 0


## GRID DITECTION FAILED  https://colab.research.google.com/drive/1Ml6IDGavNUPZzLV-4ec3Elx__rbF0fHY?usp=sharing#scrollTo=CAhpH4ojKGSa
def segment_all():
    """ FOR EVERY SUBFOLDER (LETTER TYPE), FOR EVERY IMG INSIDE OF IT, UNIFORMLY CROP IT USING PRECOMPUTED CROP_WIDTH x CROP_HEIGHT AND DO INSET CROP TO NOT INCLUDE GRIDS"""
    for page_type in os.listdir(PREP_DIR):
        inp_dir = os.path.join(PREP_DIR, page_type)
        if not os.path.isdir(inp_dir):
            continue

        letter_type = TYPE_MAP.get(page_type, page_type)
        for img_path in glob.glob(os.path.join(inp_dir, '*.png')):
            sheet = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if sheet is None:
                continue

            h, w = sheet.shape
            ys = np.linspace(0, h, NUM_ROWS + 1, dtype=int)
            xs = np.linspace(0, w, NUM_COLS + 1, dtype=int)

            base = os.path.splitext(os.path.basename(img_path))[0]
            for row in range(NUM_ROWS):
                for col in range(NUM_COLS):
                    y0, y1 = ys[row],   ys[row + 1]
                    x0, x1 = xs[col],   xs[col + 1]

                    cell = sheet[y0:y1, x0:x1]

                    y0i, y1i = y0 + MARGIN, y1 - MARGIN
                    x0i, x1i = x0 + MARGIN, x1 - MARGIN

                    if y1i > y0i and x1i > x0i:
                        inner = sheet[y0i:y1i, x0i:x1i]
                    else:
                        inner = cell

                    if is_empty_cell(inner):
                        continue

                    inner = correct_rotation(inner)

                    letter = ALPHABET[col]
                    out_dir = os.path.join(CELL_ROOT, letter_type, letter)
                    os.makedirs(out_dir, exist_ok=True)

                    out_name = f"{base}_r{row:02d}.png"
                    cv2.imwrite(os.path.join(out_dir, out_name), inner)

    print("[OK] segmentation complete")

############### FEATURE EXTRACTION ################3
def extract_block_features(img: np.ndarray,
                           grid: tuple[int,int] = (4,4)
                          ) -> np.ndarray:
    h, w = img.shape
    rows, cols = grid

    ys = np.linspace(0, h, rows+1, dtype=int)
    xs = np.linspace(0, w, cols+1, dtype=int)

    feats = []
    for r in range(rows):
        for c in range(cols):
            block = img[ys[r]:ys[r+1], xs[c]:xs[c+1]]
            feats.append(cv2.countNonZero(block))
    return np.array(feats, dtype=int)

def build_feature_dataset(cell_root: str,
                          grid: tuple[int,int] = (4,4)
                         ) -> pd.DataFrame:
    rows, cols = grid
    n_feats = rows * cols
    cols_names = [f"f_{i}" for i in range(n_feats)] + ['letter', 'page_type']
    data = []

    for page_type in os.listdir(cell_root):
        pt_dir = os.path.join(cell_root, page_type)
        if not os.path.isdir(pt_dir):
            continue

        for letter in os.listdir(pt_dir):
            letter_dir = os.path.join(pt_dir, letter)
            if not os.path.isdir(letter_dir):
                continue

            for img_path in glob.glob(os.path.join(letter_dir, '*.png')):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                feat = extract_block_features(img, grid)
                # append features + label info
                data.append(list(feat) + [letter, page_type])

    return pd.DataFrame(data, columns=cols_names)


def main():
    os.makedirs(PREP_DIR, exist_ok=True)
    os.makedirs(CELL_ROOT, exist_ok=True)

    # 1) Align & binarize each sheet
    for sub in os.listdir(BASE_DIR):
        if os.path.isdir(os.path.join(BASE_DIR, sub)):
            process_folder(sub)

    # 2) Segment into labeled cells
    segment_all()

    df = build_feature_dataset(cell_root=CELL_ROOT, grid=(4, 4))
    df.to_csv('letter_features.csv', index=False)
    print("Features saved to letter_features.csv")


if __name__ == '__main__':
    main()
