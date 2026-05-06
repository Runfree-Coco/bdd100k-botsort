import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    """纯 Python 计算 IoU"""
    x1 = np.maximum(bbox1[:, 0], bbox2[:, 0])
    y1 = np.maximum(bbox1[:, 1], bbox2[:, 1])
    x2 = np.minimum(bbox1[:, 2], bbox2[:, 2])
    y2 = np.minimum(bbox1[:, 3], bbox2[:, 3])
    inter = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1)
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def iou_distance(atracks, btracks):
    """计算两组轨迹之间的 IoU 距离矩阵"""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)))
    atlbr = np.array([t.tlbr for t in atracks])
    btlbr = np.array([t.tlbr for t in btracks])
    ious = np.zeros((len(atlbr), len(btlbr)))
    for i, a in enumerate(atlbr):
        ious[i] = iou(np.tile(a, (len(btlbr), 1)), btlbr)
    return 1 - ious

def fuse_score(cost_matrix, detections):
    """将检测置信度融合到代价矩阵中"""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    fused = iou_sim * det_scores
    return 1 - fused

def linear_assignment(cost_matrix, thresh):
    """匈牙利算法匹配"""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = []
    unmatched_cols = []
    for r in range(cost_matrix.shape[0]):
        if r not in row_ind:
            unmatched_rows.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in col_ind:
            unmatched_cols.append(c)
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
        else:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
    return matches, unmatched_rows, unmatched_cols