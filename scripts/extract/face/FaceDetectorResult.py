import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import mxnet as mx

import cv2
import numpy as np
from mxnet.module import Module

from scripts.extract.face.yunet import YuNet


def load_face_features_extraction_model(model_dir: Path) -> Module:
    sym, arg_params, aux_params = mx.model.load_checkpoint(str(model_dir) + '\\', 0)
    all_layers = sym.get_internals()
    ##if we want bach normalization features:
    # symbn1 = all_layers['bn1_output']
    symbn1 = all_layers['fc1_output']
    ctx = mx.cpu()
    model = mx.mod.Module(symbol=symbn1, context=ctx, label_names=None)
    data_shape = (1, 3) + (112, 112)
    model.bind(data_shapes=[('data', data_shape)])
    model.set_params(arg_params, aux_params)
    data = mx.nd.zeros(shape=data_shape)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    return model


def load_face_detection_model(
        model_dir: Path,
        model_file_name: str = 'face_detection_yunet_2022mar.onnx',
        input_size: Optional[Tuple[int, int]] = (300, 300)
) -> YuNet:
    face_detector = YuNet(
        model_path=str(model_dir.joinpath(model_file_name)),
        input_size=input_size,
        conf_threshold=0.5,
        nms_threshold=0.6,
        top_k=5000,
        backend_id=0,
        target_id=0)
    return face_detector


@dataclass
class FaceDetectorResult:
    face_detected: bool
    face_box: np.ndarray
    landmarks: np.ndarray


def face_detection_single_frame(img_cv2, face_detector):
    detection_success = False
    bounding_box = None
    landmarks = None
    h, w, _ = img_cv2.shape
    face_detector.setInputSize([w, h])
    try:
        mtcnn_results = face_detector.infer(img_cv2)
    except:
        return FaceDetectorResult(
            False,
            np.zeros((4), dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
        )
    if mtcnn_results is None:
        return FaceDetectorResult(
            False,
            np.zeros((4), dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
        )
    mtcnn_results = remove_small_faces(mtcnn_results)
    ##if there is no face detected, try to rotate image 90/-90 degress and detect again
    if mtcnn_results is None:
        mtcnn_results, img_cv2 = face_detection_after_rotation(img_cv2, face_detector)
        if mtcnn_results is None:
            return FaceDetectorResult(
                False,
                np.zeros((4), dtype=np.float32),
                np.zeros((5, 2), dtype=np.float32),
            )
    ##remove those detected faces with low confidence.
    list_boundingBox, list_landmarks = get_detection_results_with_high_confidence(mtcnn_results)
    if len(list_boundingBox) == 0:
        return FaceDetectorResult(
            False,
            np.zeros((4), dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
        )
    if len(list_boundingBox) == 1:
        bounding_box = list_boundingBox[0]
        landmarks = list_landmarks[0]
        landmarks_reshaped = np.array(landmarks).reshape(5, 2)
        return FaceDetectorResult(
            True,
            bounding_box,
            landmarks_reshaped,
        )
    elif len(list_boundingBox) > 1:  ## ##there maybe several faces in one frame, we need to select the largest face.
        bounding_box, landmarks = get_largest_face(list_boundingBox, list_landmarks)
        landmarks_reshaped = np.array(landmarks).reshape(5, 2)
        return FaceDetectorResult(
            True,
            bounding_box,
            landmarks_reshaped,
        )


def remove_small_faces(mtcnn_results):
    min_width = 10
    list_width = mtcnn_results[:, 2]
    needed_results = []
    index = 0
    for each_width in list_width:
        if each_width > min_width:
            needed_results.append(mtcnn_results[index])
        index += 1
    return np.array(needed_results)


def face_detection_after_rotation(img_cv2, mtcnn_detector):
    img_cv2_clockwise_rotated = cv2.rotate(img_cv2, cv2.ROTATE_90_CLOCKWISE)
    mtcnn_results = mtcnn_detector.detect_face(img_cv2_clockwise_rotated)
    if mtcnn_results is not None:
        return mtcnn_results, img_cv2_clockwise_rotated
    img_cv2_rotated_up_side_down = cv2.rotate(img_cv2_clockwise_rotated, cv2.ROTATE_90_CLOCKWISE)
    mtcnn_results = mtcnn_detector.detect_face(img_cv2_rotated_up_side_down)
    if mtcnn_results is not None:
        return mtcnn_results, img_cv2_rotated_up_side_down
    img_cv2_counter_rotated = cv2.rotate(img_cv2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    mtcnn_results = mtcnn_detector.detect_face(img_cv2_counter_rotated)
    if mtcnn_results is not None:
        return mtcnn_results, img_cv2_counter_rotated
    return None, None


def get_detection_results_with_high_confidence(mtcnn_results):
    list_boundingBox = []
    list_landmarks = []
    list_confidenceScores = mtcnn_results[:, 14]
    confidenceThreshold = 0.5
    index = 0
    if len(list_confidenceScores) == 1:
        boundingBox_key_points = mtcnn_results[0][0:4]
        list_boundingBox.append(boundingBox_key_points)
        list_landmarks.append(mtcnn_results[0][4:14])
        return list_boundingBox, list_landmarks
    for eachConfidenceScore in list_confidenceScores:
        if eachConfidenceScore > confidenceThreshold:
            boundingBox_key_points = mtcnn_results[index][0:4]
            list_boundingBox.append(boundingBox_key_points)
            list_landmarks.append(mtcnn_results[index][4:14])
        index += 1
    return list_boundingBox, list_landmarks


def get_largest_face(list_boundingBox, list_landmarks):
    list_width = []
    for eachBoundxingBox in list_boundingBox:
        list_width.append(eachBoundxingBox[2])
    maxValue = max(list_width)
    index_maxvalue = list_width.index(maxValue)
    return list_boundingBox[index_maxvalue], list_landmarks[index_maxvalue]
