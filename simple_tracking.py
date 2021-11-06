import cv2
import numpy as np


def calc_center(box):
    """
    boxの中心x座標を返す

    Parameters
    ----------
    box: np.ndarray
        [x1, y1, x2 y2]となっている配列

    Returns
    ----------
    center: int
        boxの中心x座標
    """
    center = box[0] + (box[2] - box[0]) // 2
    return center


class MultiObjectTracker:
    def __init__(self, threshold_dx=30, self.=3):
        self.detections = []
        self.threshold_dx = threshold_dx
        self.threshold_cnt = threshold_cnt

    def update(self, boxes):
        """
        検出したboxesをself.detectionsに対応させる
        新規の場合はself.detectionsに追加する
        self.detections内の全てのDetectionに対して、Detection.predict_centerとDetection.cntを更新する
        self.detections内の全てのDetectionに対して、Detection.cnt > threshold_cntとなっているDetectionを削除する

        Parameters
        ----------
        boxes: np.ndarray
            検出した配列
        """

        # is_updatedを0に初期化
        for detection in self.detections:
            detection.is_updated = 0

        # dx_matrixの作成 検出したboxesと既に検出しているself.detectionsとの距離を全ての組み合わせで求める
        dx_matrix = np.full((len(boxes), len(self.detections)), self.threshold_dx)
        for i, box in enumerate(boxes):
            center = calc_center(box)
            for j, detection in enumerate(self.detections):
                dx = abs(center - detection.predict_center)
                dx_matrix[i][j] = dx

        # 新規のboxes 既存のself.detectionsを更新していく際に削除し、残ったものが新規となる
        new_boxes_idx = {idx for idx in range(len(boxes))}

        # 既存のself.detectionsを更新
        if dx_matrix.size:
            for _ in range(len(boxes)):
                min_idx = np.unravel_index(np.argmin(dx_matrix), dx_matrix.shape)
                if dx_matrix[min_idx] < self.threshold_dx:
                    box_idx, detection_idx = min_idx[0], min_idx[1]
                    self.detections[detection_idx].update(boxes[box_idx])
                    new_boxes_idx.discard(box_idx)
                    dx_matrix[box_idx, :] = self.threshold_dx
                    dx_matrix[:, detection_idx] = self.threshold_dx
                else:
                    break

        # 新規のboxesをself.detectionsに追加
        for box_idx in new_boxes_idx:
            self.detections.append(Detection(boxes[box_idx]))

        # self.detections内の、Detection.cnt > threshold_cntとなっているDetectionを削除
        del_idx = []
        for i, detection in enumerate(self.detections):
            detection.cnt += 1
            detection.predict_center = calc_center(detection.pre_box) + detection.calc_dx() * detection.cnt
            if detection.cnt > self.threshold_cnt:
                del_idx.append(i)
        for i in reversed(del_idx):
            self.detections.pop(i)

    def draw(self, frame):
        """
        self.detectionsのboxとlabelを描画

        Parameters
        ----------
        frame: np.ndarray
            描画元
        """
        for detection in self.detections:
            if not detection.is_updated:  # 現在のフレームで検出されなかった場合はcontinue
                continue
            label = detection.label
            color = [ord(c) * ord(c) % 256 for c in label[:3]]
            box = [int(i) for i in detection.pre_box]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness=4)
            cv2.putText(frame, label[:5], (box[0], box[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color)


class Detection:
    def __init__(self, box):
        self.label = str(np.random.randint(10000, 100000))[:5]
        self.past_dx = []
        self.pre_box = box
        self.predict_center = calc_center(box)
        self.cnt = 0
        self.is_updated = 1

    def update(self, box):
        center = calc_center(box)
        dx = center - calc_center(self.pre_box)
        self.past_dx.append(dx)
        if len(self.past_dx) > 3:
            self.past_dx.pop(0)

        self.pre_box = box

        self.cnt = 0

        self.is_updated = 1

    def calc_dx(self):
        """
        過去3つのdxの平均を返す
        """

        if self.past_dx:
            return sum(self.past_dx) / len(self.past_dx)
        else:
            return 0
