from typing import List, Dict, Tuple
import random as rd

import cv2
import numpy as np
from mxnet.module import Module
from tqdm import tqdm

from scripts.Subject import Subject
from scripts.extract.face import yunet_scripts
from scripts.extract.face.face_extract import extract_face
from scripts.extract.face.yunet import YuNet


def bench_merged_frames_efficiencies(
        subjects: List[Subject],
        model_feature_extraction: Module,
        min_threshold: int,
        confidence_percentage: float
) -> None:
    unsatisfied = dict()
    suspicious = dict()
    confidence_threshold = min_threshold + confidence_percentage
    total_it = len(subjects) * (len(subjects) - 1)
    progress_bar = tqdm(total=total_it, desc=" Checking similarity", unit=" Subject ")

    for subject_src in subjects:
        reference_faces = [img for img in subject_src.frame.face.frames_dir().glob("*.png")]
        for subject_dst in subjects:
            if subject_src == subject_dst:
                continue
            merged_frames = [img for img in subject_src.frame.face.frames_dir_from(subject_dst.id()).glob("*.png")]
            for frame in merged_frames:
                face = cv2.imread(str(frame))
                rd_reference_face = reference_faces[rd.randint(0, len(reference_faces) - 1)]
                rd_reference_face = cv2.imread(str(rd_reference_face))
                features_face = yunet_scripts.extractionFeature_cropped_image_without_saving(
                    face, model_feature_extraction
                )
                features_ref = yunet_scripts.extractionFeature_cropped_image_without_saving(
                    rd_reference_face, model_feature_extraction
                )
                score = calculate_similarity_score(features_ref, features_face)
                del face
                del rd_reference_face
                key = (subject_src.id(), subject_dst.id())
                if score <= min_threshold:
                    if key in unsatisfied.keys():
                        item = list(unsatisfied[key])
                        item[0] += 1
                        unsatisfied[key] = tuple(item)
                    else:
                        unsatisfied[key] = (1, len(merged_frames))
                elif score <= confidence_threshold:
                    if key in suspicious.keys():
                        item = list(suspicious[key])
                        item[0] += 1
                        suspicious[key] = tuple(item)
                    else:
                        suspicious[key] = (1, len(merged_frames))
            progress_bar.update(1)

    progress_bar.close()

    bench_results(unsatisfied, suspicious, min_threshold, confidence_threshold)


def calculate_similarity_score(feature_vector_reference, feature_vector_probe):
    similarity_score = np.inner(feature_vector_reference, feature_vector_probe)
    if similarity_score > 1.0:
        similarity_score = 1.0
    if similarity_score < -1.0:
        similarity_score = -1.0
    return similarity_score


def bench_results(
        unsatisfied: Dict[Tuple[int, int], Tuple[int, int]],
        suspicious: Dict[Tuple[int, int], Tuple[int, int]],
        threshold: float,
        sus_threshold: float
) -> None:
    if len(suspicious) == 0:
        print("No suspicious frames have been generated.")
    else:
        print(f" -- Suspicious subject: Threshold <= {sus_threshold} --")
        for key, value in suspicious.items():
            print(f" > [SRC: {key[0]}, DST: {key[1]}]: found {value[0]} suspicious frames over {value[1]} generated "
                  f"frames.")
    if len(unsatisfied) == 0:
        print("No unsatisfied frames have been generated.")
    else:
        print(f" -- Unsatisfied subject Threshold <= {threshold} --")
        for key, value in unsatisfied.items():
            print(f" > [SRC: {key[0]}, DST: {key[1]}]: found {value[0]} unsatisfied frames over {value[1]} generated "
                  f"frames.")
