from typing import List, Dict, Tuple
import random as rd

import numpy as np
from mxnet.module import Module

from scripts.Subject import Subject
from scripts.extract.face import yunet_scripts
from scripts.extract.face.face_extract import extract_face
from scripts.extract.face.yunet import YuNet


def bench_merged_frames_efficiencies(
        subjects: List[Subject],
        face_detector_model: YuNet,
        model_feature_extraction: Module,
        max_shape: int,
        min_threshold: int,
        confidence_percentage: float
) -> None:
    unsatisfied = dict()
    suspicious = dict()
    confidence_threshold = min_threshold + (confidence_percentage * min_threshold / 100)

    for subject_src in subjects:
        reference_faces = [img for img in subject_src.frame.original.frames_dir().iterdir()]
        for subject_dst in subjects:
            if subject_src == subject_dst:
                continue
            merged_frames = [img for img in subject_src.frame.merged.frames_dir_from(subject_dst.id()).iterdir()]
            for frame in merged_frames:
                face = extract_face(frame, face_detector_model, max_shape)
                rd_reference_face = extract_face(
                    reference_faces[rd.randint(0, len(reference_faces))],
                    face_detector_model,
                    max_shape
                )
                features_face = yunet_scripts.extractionFeature_cropped_image_without_saving(
                    face, model_feature_extraction
                )
                features_ref = yunet_scripts.extractionFeature_cropped_image_without_saving(
                    rd_reference_face, model_feature_extraction
                )
                score = calculate_similarity_score(features_ref, features_face)
                key = (subject_src.id(), subject_dst.id())
                if score <= min_threshold:
                    if key in unsatisfied.keys():
                        unsatisfied[key][0] += 1
                    else:
                        unsatisfied[key] = (0, len(merged_frames))
                elif score <= confidence_threshold:
                    if key in suspicious.keys():
                        suspicious[key][0] += 1
                    else:
                        suspicious[key] = (0, len(merged_frames))

    bench_results(unsatisfied, suspicious)


def calculate_similarity_score(feature_vector_reference, feature_vector_probe):
    similarity_score = np.inner(feature_vector_reference, feature_vector_probe)
    if similarity_score > 1.0:
        similarity_score = 1.0
    if similarity_score < -1.0:
        similarity_score = -1.0
    return similarity_score


def bench_results(
        unsatisfied: Dict[Tuple[int, int], Tuple[int, int]],
        suspicious: Dict[Tuple[int, int], Tuple[int, int]]
) -> None:
    if len(suspicious) == 0:
        print("No suspicious frames have been generated.")
    else:
        print(" -- Suspicious subject --")
        for key, value in suspicious.items():
            print(f" > [SRC: {key[0]}, DST: {key[1]}]: found {value[0]} suspicious frames over {value[1]} generated "
                  f"frames.")
    if len(unsatisfied) == 0:
        print("No unsatisfied frames have been generated.")
    else:
        print(" -- Suspicious subject --")
        for key, value in unsatisfied.items():
            print(f" > [SRC: {key[0]}, DST: {key[1]}]: found {value[0]} unsatisfied frames over {value[1]} generated "
                  f"frames.")
