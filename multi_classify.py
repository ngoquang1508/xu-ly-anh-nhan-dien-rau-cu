import cv2
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# ============================================================
# CẤU HÌNH
# ============================================================

DB_PATH      = "vegetable_database.json"
DATASET_PATH = r"D:\xla\nhan_dien_rau_cu\archive\Vegetable Images"
TEST_PATH    = os.path.join(DATASET_PATH, "test")

IMG_SIZE    = (128, 128)
LBP_RADIUS  = 3
LBP_POINTS  = 24
K_NEIGHBORS = 7

COLORS = [
    (0,255,0),(0,128,255),(255,0,128),(255,255,0),(0,255,255),
    (255,0,255),(128,255,0),(0,80,255),(255,128,0),(80,255,80),
]

# ============================================================
# NHÃN TIẾNG VIỆT KHÔNG DẤU
# ============================================================

VIETNAMESE_LABELS = {
    "Bean": "Dau",
    "Bitter_Gourd": "Kho qua",
    "Bottle_Gourd": "Bau",
    "Brinjal": "Ca tim",
    "Broccoli": "Bong cai xanh",
    "Cabbage": "Bap cai",
    "Capsicum": "Ot chuong",
    "Carrot": "Ca rot",
    "Cauliflower": "Sup lo",
    "Cucumber": "Dua leo",
    "Papaya": "Du du",
    "Potato": "Khoai tay",
    "Pumpkin": "Bi do",
    "Radish": "Cu cai",
    "Tomato": "Ca chua"
}

# ============================================================
# TRÍCH XUẤT VECTOR
# ============================================================

def extract_vector(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.normalize(cv2.calcHist([hsv],[0],None,[36],[0,180]),None).flatten()
    hist_s = cv2.normalize(cv2.calcHist([hsv],[1],None,[32],[0,256]),None).flatten()

    color_hist = np.concatenate([hist_h, hist_s])

    color_mean = np.array([
        np.mean(hsv[:,:,0])/180,
        np.mean(hsv[:,:,1])/255,
        np.mean(hsv[:,:,2])/255,
        np.std(hsv[:,:,0])/180,
        np.std(hsv[:,:,1])/255,
        np.std(hsv[:,:,2])/255,
    ])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')

    lbp_hist,_ = np.histogram(
        lbp.ravel(),
        bins=LBP_POINTS+2,
        range=(0,LBP_POINTS+2),
        density=True
    )

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:

        c = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(c)
        peri = cv2.arcLength(c,True)

        x,y,w,h = cv2.boundingRect(c)

        shape = np.array([
            float(4*np.pi*area/peri**2) if peri>0 else 0,
            float(w)/h if h>0 else 1.0,
            float(area)/(w*h) if w*h>0 else 0
        ])
    else:
        shape = np.array([0.0,1.0,0.0])


    b = img[:,:,0].astype(float)
    g = img[:,:,1].astype(float)
    r = img[:,:,2].astype(float)

    color_extra = np.array([
        np.mean((r>g+40)&(r>b+40)) * 3.0,
        np.mean((r>b+40)&(g>b+20)) * 3.0,
        np.mean((g>r+20)&(g>b+20)) * 3.0,
        float(np.mean(r-g))/255.0 * 3.0,
        float(np.mean(r-b))/255.0 * 3.0,
    ])

    return np.concatenate([
        color_hist,
        color_mean,
        lbp_hist,
        shape,
        color_extra
    ])


# ============================================================
# KNN
# ============================================================

def classify_knn(vec, db, k=K_NEIGHBORS):

    distances = {}

    for veg, mat in db.items():

        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10

        sims = mat.dot(vec) / (norms.flatten() * (np.linalg.norm(vec)+1e-10))

        distances[veg] = float(np.mean(np.sort(sims)[-k:]))

    best = max(distances, key=distances.get)

    return best, distances[best]


# ============================================================
# SEGMENTATION
# ============================================================

def segment_objects(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)

    _, thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY_INV)

    k_close = np.ones((15,15), np.uint8)
    k_open  = np.ones((5,5), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k_open)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = img.shape[0]*img.shape[1]*0.01

    regions = []

    for cnt in contours:

        if cv2.contourArea(cnt) >= min_area:

            x,y,w,h = cv2.boundingRect(cnt)

            regions.append((x,y,w,h,cnt))

    return regions


# ============================================================
# PHÂN LOẠI + GÁN NHÃN
# ============================================================

def classify_and_label(img, db):

    result = img.copy()

    regions = segment_objects(img)

    print(f"\nTim thay {len(regions)} vat the")

    predictions = []

    for i,(x,y,w,h,cnt) in enumerate(regions):

        roi = img[y:y+h, x:x+w]

        roi_r = cv2.resize(roi, IMG_SIZE)

        vec = extract_vector(roi_r)

        label,score = classify_knn(vec, db)

        label_vi = VIETNAMESE_LABELS.get(label,label)

        color = COLORS[i % len(COLORS)]

        cv2.drawContours(result,[cnt],-1,color,2)

        cv2.rectangle(result,(x,y),(x+w,y+h),color,2)

        text = f"{label_vi} ({score:.2f})"

        (tw,th),_ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        cv2.rectangle(result,(x,y-th-10),(x+tw+8,y),color,-1)

        cv2.putText(
            result,
            text,
            (x+4,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        predictions.append((label_vi,score,x,y,w,h))

        print(f"Vat the {i+1}: {label_vi:<20} score={score:.3f}")

    # ===== HIỂN THỊ BẰNG MATPLOTLIB =====

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12,8))
    plt.imshow(result_rgb)
    plt.axis("off")
    plt.title("Ket qua nhan dien rau cu")
    plt.show()

    return predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import sys

    with open(DB_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    db = {
        veg: np.array(entry["vectors"], dtype=np.float32)
        for veg, entry in raw.items()
    }

    print(
        f"Database: {len(db)} loai | "
        f"{sum(v.shape[0] for v in db.values())} vectors"
    )

    if len(sys.argv) > 1:

        img_path = sys.argv[1]

        img = cv2.imread(img_path)

        if img is None:

            print("Khong doc duoc anh")

            sys.exit(1)

        preds = classify_and_label(img, db)

        print("Hoan thanh!")