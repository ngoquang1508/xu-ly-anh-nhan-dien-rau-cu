import cv2
import numpy as np
import json
import os
from skimage.feature import local_binary_pattern

# ============================================================
# CẤU HÌNH
# ============================================================

DATASET_PATH = r"D:\xla\nhan_dien_rau_cu\archive\Vegetable Images"
TRAIN_PATH   = os.path.join(DATASET_PATH, "train")
OUTPUT_DB    = "vegetable_database.json"

NUM_SAMPLES  = 100
IMG_SIZE     = (128, 128)
LBP_RADIUS   = 3
LBP_POINTS   = 24


# ============================================================
# TRÍCH XUẤT VECTOR  (v4: thêm red/RGB ratio)
# ============================================================

def extract_vector(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Histogram H (36) + S (32) = 68 chiều
    hist_h = cv2.normalize(cv2.calcHist([hsv],[0],None,[36],[0,180]),None).flatten()
    hist_s = cv2.normalize(cv2.calcHist([hsv],[1],None,[32],[0,256]),None).flatten()
    color_hist = np.concatenate([hist_h, hist_s])

    # 2. Mean + std HSV = 6 chiều
    color_mean = np.array([
        np.mean(hsv[:,:,0])/180, np.mean(hsv[:,:,1])/255, np.mean(hsv[:,:,2])/255,
        np.std(hsv[:,:,0])/180,  np.std(hsv[:,:,1])/255,  np.std(hsv[:,:,2])/255,
    ])

    # 3. Texture LBP = 26 chiều
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist,_ = np.histogram(lbp.ravel(), bins=LBP_POINTS+2,
                               range=(0,LBP_POINTS+2), density=True)

    # 4. Hình dạng = 3 chiều
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c); peri = cv2.arcLength(c, True)
        x,y,w,h = cv2.boundingRect(c)
        shape = np.array([
            float(4*np.pi*area/peri**2) if peri>0 else 0,
            float(w)/h if h>0 else 1.0,
            float(area)/(w*h) if w*h>0 else 0,
        ])
    else:
        shape = np.array([0.0, 1.0, 0.0])

    # 5. *** MỚI: Đặc trưng màu đỏ riêng = 5 chiều ***
    b = img[:,:,0].astype(float)
    g = img[:,:,1].astype(float)
    r = img[:,:,2].astype(float)

    # Tỉ lệ pixel "đỏ mạnh" trong ảnh (R > G+40 và R > B+40)
    red_mask    = ((r > g + 40) & (r > b + 40)).astype(float)
    red_ratio   = np.mean(red_mask)                   # [0..1]

    # Tỉ lệ pixel "cam/vàng" (R > B+40 và G > B+20)
    orange_mask = ((r > b + 40) & (g > b + 20)).astype(float)
    orange_ratio = np.mean(orange_mask)

    # Tỉ lệ pixel "xanh lá"
    green_mask  = ((g > r + 20) & (g > b + 20)).astype(float)
    green_ratio = np.mean(green_mask)

    # Mean R-G và R-B (normalize về [-1,1])
    rg_diff = float(np.mean(r - g)) / 255.0
    rb_diff = float(np.mean(r - b)) / 255.0

    # Nhân trọng số 3x để nổi bật hơn trong vector
    color_extra = np.array([
        red_ratio    * 3.0,
        orange_ratio * 3.0,
        green_ratio  * 3.0,
        rg_diff      * 3.0,
        rb_diff      * 3.0,
    ])

    # Tổng: 68 + 6 + 26 + 3 + 5 = 108 chiều
    return np.concatenate([color_hist, color_mean, lbp_hist, shape, color_extra])


# ============================================================
# XÂY DỰNG DATABASE
# ============================================================

def build_database():
    print("=" * 60)
    print("   XÂY DỰNG DATABASE  (v4 — thêm red/color ratio)")
    print("=" * 60)

    veg_classes = sorted([
        d for d in os.listdir(TRAIN_PATH)
        if os.path.isdir(os.path.join(TRAIN_PATH, d))
    ])
    print(f"\n📂 {len(veg_classes)} loại: {', '.join(veg_classes)}\n")

    database = {}

    for veg in veg_classes:
        veg_dir   = os.path.join(TRAIN_PATH, veg)
        img_files = [
            f for f in os.listdir(veg_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:NUM_SAMPLES]

        vectors = []
        for fname in img_files:
            img = cv2.imread(os.path.join(veg_dir, fname))
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE)
            vectors.append(extract_vector(img).tolist())

        if not vectors:
            print(f"  ⚠️  {veg}: bỏ qua")
            continue

        # In thống kê đặc trưng đỏ để kiểm tra
        mat = np.array(vectors)
        red_avg    = np.mean(mat[:, -5]) / 3.0   # bỏ trọng số để đọc
        orange_avg = np.mean(mat[:, -4]) / 3.0
        green_avg  = np.mean(mat[:, -3]) / 3.0

        database[veg] = {
            "label":       veg,
            "num_samples": len(vectors),
            "vectors":     vectors,
        }
        print(f"  ✅ {veg:<20} {len(vectors)} vec  "
              f"red={red_avg:.3f}  orange={orange_avg:.3f}  green={green_avg:.3f}")

    with open(OUTPUT_DB, "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False)

    size_mb = os.path.getsize(OUTPUT_DB) / 1024 / 1024
    print(f"\n💾 Đã lưu: {OUTPUT_DB}  ({len(database)} loại, {size_mb:.1f} MB)")
    print("✅ XONG! Chạy classify.py để đánh giá.")


if __name__ == "__main__":
    build_database()
