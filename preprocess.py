import cv2
from pathlib import Path
from tqdm import tqdm

SRC = Path("train")
DST = Path("train_preprocessed")
DST.mkdir(exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for src_img in tqdm(list(SRC.glob("*.png"))):
    img = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)

    # 1. Denoise (bilateral preserves edges better than Gaussian for X-rays)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. CLAHE (histogram equalization)
    img = clahe.apply(img)

    # 3. Contrast stretch
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # YOLO expects 3 channels
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(DST / src_img.name), img)