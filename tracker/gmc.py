"""
GMC (Global Motion Compensation) — 稀疏光流仿射估算
每帧返回 2x3 仿射矩阵，用于补偿相机运动对 Kalman 预测的影响。
"""

import cv2
import numpy as np


class GMC:
    def __init__(self, downscale: int = 2):
        self.downscale = max(1, downscale)
        self.prev_gray = None
        self.prev_kps = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )

    def reset(self):
        self.prev_gray = None
        self.prev_kps = None

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        输入: BGR 图像 (H, W, 3)
        返回: 2x3 仿射矩阵；若无法估算则返回单位矩阵
        """
        h, w = img.shape[:2]
        sh, sw = h // self.downscale, w // self.downscale
        gray = cv2.cvtColor(cv2.resize(img, (sw, sh)), cv2.COLOR_BGR2GRAY)

        identity = np.eye(2, 3, dtype=np.float32)

        if self.prev_gray is None or self.prev_kps is None or len(self.prev_kps) < 4:
            self.prev_gray = gray
            self.prev_kps = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return identity

        curr_kps, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_kps, None, **self.lk_params
        )

        if curr_kps is None or status is None:
            self.prev_gray = gray
            self.prev_kps = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return identity

        good_prev = self.prev_kps[status.ravel() == 1]
        good_curr = curr_kps[status.ravel() == 1]

        if len(good_prev) < 4:
            self.prev_gray = gray
            self.prev_kps = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return identity

        M, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        self.prev_gray = gray
        self.prev_kps = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)

        if M is None:
            return identity

        # 坐标是在缩放图上估算的，需要还原到原始尺度（平移部分 * downscale，旋转/缩放不变）
        M[0, 2] *= self.downscale
        M[1, 2] *= self.downscale
        return M.astype(np.float32)
