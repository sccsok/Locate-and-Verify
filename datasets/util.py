import cv2
import dlib
from collections import OrderedDict
import numpy as np
from imutils import face_utils
import torch


class FaceLandmark:
    def __init__(self, lmk_path='./datasets/shape_predictor_68_face_landmarks.dat', ref='mouth'):
        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 35)),
            ("jaw", (0, 17))
        ])
        self.ref = ref
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(lmk_path)

    def get_mask(self, image):
        h, w, c = image.shape

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        mask = np.zeros_like(image)

        for face in faces:  # (x, y, w, h)
            # cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 5)
            shape = face_utils.shape_to_np(self.predictor(image, face))
            if self.ref in ['mouth', 'nose']:
                i, j = self.FACIAL_LANDMARKS_IDXS[self.ref]
                ref_co = shape[i: j]
                xmin, xmax = min(ref_co[:, 0]) - 5, max(ref_co[:, 0]) + 5
                ymin, ymax = min(ref_co[:, 1]) - 5, max(ref_co[:, 1]) + 5
                mask[ymin:ymax, xmin:xmax, :] = 255
            elif self.ref == 'eyes':
                ri, rj = self.FACIAL_LANDMARKS_IDXS['right_eye']
                li, lj = self.FACIAL_LANDMARKS_IDXS['left_eye']
                rref_co, lref_co = shape[ri: rj], shape[li: lj]
                xmin, xmax = min(rref_co[:, 0]) - 5, max(lref_co[:, 0]) + 5
                ymin, ymax = min(min(rref_co[:, 1]), min(lref_co[:, 1])) - 5, max(max(rref_co[:, 1]), max(lref_co[:, 1])) + 5
                mask[ymin:ymax, xmin:xmax, :] = 255
                
        return  mask


# fcm = FaceLandmark()
# mask = fcm.get_mask(image)