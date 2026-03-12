import cv2
import numpy as np
import json
import os
import sys
from skimage.feature import local_binary_pattern

# ============================================================
# CẤU HÌNH
# ============================================================

DB_PATH      = "vegetable_database.json"
DATASET_PATH = r"D:\xla\nhan_dien_rau_cu\archive\Vegetable Images"
TEST_PATH    = os.path.join(DATASET_PATH, "test")
OUTPUT_DIR   = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = (128, 128)
LBP_RADIUS  = 3
LBP_POINTS  = 24
K_NEIGHBORS = 7


# ============================================================
# TRÍCH XUẤT VECTOR  (v4: 108 chiều)
# ============================================================

def extract_vector(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.normalize(cv2.calcHist([hsv],[0],None,[36],[0,180]),None).flatten()
    hist_s = cv2.normalize(cv2.calcHist([hsv],[1],None,[32],[0,256]),None).flatten()
    color_hist = np.concatenate([hist_h, hist_s])

    color_mean = np.array([
        np.mean(hsv[:,:,0])/180, np.mean(hsv[:,:,1])/255, np.mean(hsv[:,:,2])/255,
        np.std(hsv[:,:,0])/180,  np.std(hsv[:,:,1])/255,  np.std(hsv[:,:,2])/255,
    ])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist,_ = np.histogram(lbp.ravel(), bins=LBP_POINTS+2,
                               range=(0,LBP_POINTS+2), density=True)

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

    # Đặc trưng màu đỏ/cam/xanh riêng (x3 trọng số)
    b = img[:,:,0].astype(float)
    g = img[:,:,1].astype(float)
    r = img[:,:,2].astype(float)
    red_ratio    = np.mean((r > g+40) & (r > b+40))
    orange_ratio = np.mean((r > b+40) & (g > b+20))
    green_ratio  = np.mean((g > r+20) & (g > b+20))
    rg_diff      = float(np.mean(r - g)) / 255.0
    rb_diff      = float(np.mean(r - b)) / 255.0

    color_extra = np.array([
        red_ratio    * 3.0,
        orange_ratio * 3.0,
        green_ratio  * 3.0,
        rg_diff      * 3.0,
        rb_diff      * 3.0,
    ])

    return np.concatenate([color_hist, color_mean, lbp_hist, shape, color_extra])


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database(db_path):
    with open(db_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    db = {veg: np.array(entry["vectors"], dtype=np.float32)
          for veg, entry in raw.items()}
    total = sum(v.shape[0] for v in db.values())
    print(f"✅ Database: {len(db)} loại  |  {total} vectors tổng cộng\n")
    return db


# ============================================================
# KNN CLASSIFICATION
# ============================================================

def classify_knn(vec, db, k=K_NEIGHBORS):
    distances = {}
    for veg, mat in db.items():
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        sims  = mat.dot(vec) / (norms.flatten() * (np.linalg.norm(vec)+1e-10))
        distances[veg] = float(np.mean(np.sort(sims)[-k:]))
    best = max(distances, key=distances.get)
    top3 = sorted(distances.items(), key=lambda x: -x[1])[:3]
    return best, distances[best], top3


# ============================================================
# ĐÁNH GIÁ TOÀN BỘ TẬP TEST
# ============================================================

def evaluate_test_set(db):
    print("=" * 60)
    print("   ĐÁNH GIÁ TRÊN TẬP TEST  (v4)")
    print("=" * 60)

    veg_classes = sorted([
        d for d in os.listdir(TEST_PATH)
        if os.path.isdir(os.path.join(TEST_PATH, d))
    ])
    print(f"📂 {len(veg_classes)} loại rau củ\n")

    total_correct = 0
    total_count   = 0

    for veg in veg_classes:
        veg_dir   = os.path.join(TEST_PATH, veg)
        img_files = [f for f in os.listdir(veg_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files: continue

        correct = 0
        for fname in img_files:
            img = cv2.imread(os.path.join(veg_dir, fname))
            if img is None: continue
            vec  = extract_vector(cv2.resize(img, IMG_SIZE))
            pred, _, _ = classify_knn(vec, db)
            if pred == veg:
                correct += 1

        count = len(img_files)
        acc   = correct / count * 100 if count > 0 else 0
        total_correct += correct
        total_count   += count

        status = "✅" if acc >= 60 else "⚠️ "
        print(f"  {status} {veg:<20} {correct:>3}/{count:<3}  ({acc:.1f}%)")

    overall = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"\n{'─'*60}")
    print(f"  📊 TỔNG: {total_correct}/{total_count}  Accuracy = {overall:.1f}%")


# ============================================================
# PHÂN LOẠI 1 ẢNH + GÁN NHÃN
# ============================================================

def classify_and_show(img_path, db):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không đọc được: {img_path}")
        return

    vec  = extract_vector(cv2.resize(img, IMG_SIZE))
    pred, score, top3 = classify_knn(vec, db)

    print(f"\n🖼️  {os.path.basename(img_path)}")
    print(f"  🏆 Kết quả : {pred}  (score={score:.3f})")
    print(f"  Top 3:")
    for i, (lbl, s) in enumerate(top3, 1):
        bar = '█' * int(s * 30)
        print(f"    {i}. {lbl:<20} {s:.3f}  {bar}")

    out  = img.copy()
    text = f"{pred} ({score:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (0, 0), (tw+16, th+16), (0,0,0), -1)
    cv2.putText(out, text, (8, th+8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,100), 2)

    out_path = os.path.join(OUTPUT_DIR, "result_" + os.path.basename(img_path))
    cv2.imwrite(out_path, out)
    print(f"\n  💾 Đã lưu: {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    db = load_database(DB_PATH)
    if len(sys.argv) > 1:
        classify_and_show(sys.argv[1], db)
    else:
        evaluate_test_set(db)
