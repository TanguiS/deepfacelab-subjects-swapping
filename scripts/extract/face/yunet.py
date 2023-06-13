# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import cv2 as cv


class YuNet:
    def __init__(
            self,
            model_path,
            input_size=[320, 320],
            conf_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=0,
            target_id=0
    ) -> None:
        self._modelPath = model_path
        self._inputSize = tuple(input_size)  # [w, h]
        self._confThreshold = conf_threshold
        self._nmsThreshold = nms_threshold
        self._topK = top_k
        self._backendId = backend_id
        self._targetId = target_id

        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend_id):
        self._backendId = backend_id
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setTarget(self, target_id):
        self._targetId = target_id
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return faces[1]
