"""Example usage: 
    python draw2.py gt40761.txt MVI_40761.txt img1 191 4

    python draw2.py gt_file.txt prediction_putr.txt img_folder start_frame number_of_frames
"""

import cv2, numpy as np, sys
from scipy.optimize import linear_sum_assignment

ground_truth_file = sys.argv[1]
prediction_file = sys.argv[2]
img_folder = sys.argv[3]
start_frame = int(sys.argv[4])
n_frames = int(sys.argv[5])

read_txt = lambda p: np.loadtxt(p, delimiter=",", usecols=(0, 1, 2, 3, 4, 5))

gt, pr = read_txt(ground_truth_file), read_txt(prediction_file)


"""Computes IoU between a and b"""
def iou(a, b):
    iw = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    ih = max(0, min(a[2] + a[4], b[2] + b[4]) - max(a[2], b[2]))
    i = iw * ih
    return i / (a[3] * a[4] + b[3] * b[4] - i)

"""Draws box"""
def box(im, r, col, lab, dy):
    x, y, w, h = [int(v) for v in r[1:5]]
    cv2.rectangle(im, (x, y), (x + w, y + h), col, 2)
    cv2.putText(im, lab, (x, y + dy), 0, 0.45, col, 1, cv2.LINE_AA)

"""Program loop"""
for frame in range(start_frame, start_frame + n_frames):
    im = cv2.imread(f"{img_folder}/{frame:06d}.jpg")

    g = gt[gt[:, 0] == frame][:, 1:]
    p = pr[pr[:, 0] == frame][:, 1:]
    
    correct_preds = set()

    if len(g) and len(p):
        M = np.array([[iou(a, b) for b in p] for a in g])
        correct_preds = {i for i, j in zip(*linear_sum_assignment(-M)) if M[i, j] >= 0.5}

    for i, r in enumerate(g):
        box(im, r, (0, 200, 0) if i in correct_preds else (0, 215, 255),
            f"GT{int(r[0])}" + ("" if i in correct_preds else " MISS"), -5)
        
    for r in p:
        box(im, r, (0, 0, 235), f"ID{int(r[0])}", int(r[4]) + 14)

    cv2.putText(im, f"f{frame}  GT {len(g)}  pred {len(p)}  TP {len(correct_preds)}  FN {len(g)-len(correct_preds)}",
                (8, 22), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite(f"out_{frame:05d}.jpg", im)
