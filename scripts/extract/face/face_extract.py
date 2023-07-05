from pathlib import Path
from typing import List

import cv2
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from scripts.Subject import Subject
from scripts.extract.face.FaceDetectorResult import face_detection_single_frame, FaceDetectorResult
from scripts.extract.face.face_align import norm_crop, rezize_from_max_length
from scripts.extract.face.yunet import YuNet


def pre_processing_image(
        detection_result: FaceDetectorResult,
        source_image: ndarray,
        ratio_value: float = 1.0,
        max_shape: int = 112
) -> ndarray:
    landmarks_reshaped = np.array(detection_result.landmarks / ratio_value).reshape(5, 2)
    cropped_image = norm_crop(source_image, landmarks_reshaped, image_size=max_shape, mode='arcface')
    return cropped_image


def extract_face(image_path: Path, face_detector_model: YuNet, max_shape: int = 112) -> ndarray:
    image_cv2_probe = cv2.imread(str(image_path))
    if image_cv2_probe is None:
        raise ValueError
    image_cv2_probe_resized, ratio_value_probe = rezize_from_max_length(image_cv2_probe, max_shape)
    face_detection: FaceDetectorResult = face_detection_single_frame(image_cv2_probe_resized, face_detector_model)
    return pre_processing_image(face_detection, image_cv2_probe, ratio_value_probe, max_shape)


def save_extracted_face(image: ndarray, output_dir: Path, output_name: str) -> None:
    cv2.imwrite(str(output_dir.joinpath(output_name)), image)


def is_valid_frame(frame: Path) -> bool:
    return frame.exists() and frame.is_file() and frame.suffix == ".png"


def are_subjects_extractable(subject_src: Subject, subject_dst: Subject, max_shape: int) -> bool:
    return subject_src != subject_dst and not subject_src.frame.face.is_extracting_done_from(subject_dst.id(),
                                                                                             max_shape)


def is_subject_extractable(subject_src: Subject, max_shape: int) -> bool:
    return not subject_src.frame.face.is_extracting_done(max_shape)


def extract_face_from_list(
        face_detector_model: YuNet,
        max_shape: int,
        merged_frames: List[Path],
        output_dir: Path
) -> None:
    for frame in merged_frames:
        extracted_face = extract_face(frame, face_detector_model, max_shape)
        save_extracted_face(
            extracted_face,
            output_dir,
            f"{max_shape}_{frame.name}"
        )


def extract_face_from_subject(subjects: List[Subject], face_detector_model: YuNet, max_shape: int = 112):
    total_it = len(subjects) * (len(subjects) - 1) + len(subjects)
    progress_bar = tqdm(total=total_it, desc=" Extracting faces from subjects", unit=" Subject")
    for subject_src in subjects:
        for subject_dst in subjects:
            if not are_subjects_extractable(subject_src, subject_dst, max_shape):
                if subject_src == subject_dst:
                    continue
                progress_bar.update(1)
                continue

            merged_frames = [
                frame for frame in subject_src.frame.merged.frames_dir_from(subject_dst.id()).iterdir() if
                is_valid_frame(frame)
            ]
            extract_face_from_list(face_detector_model, max_shape, merged_frames,
                                   subject_src.frame.face.frames_dir_from(subject_dst.id()))
            subject_src.frame.face.extracting_done_from(subject_dst.id(), max_shape)
            progress_bar.update(1)

        if not is_subject_extractable(subject_src, max_shape):
            progress_bar.update(1)
            continue

        original_frames = [
            frame for frame in subject_src.frame.original.frames_dir().iterdir() if is_valid_frame(frame)
        ]
        extract_face_from_list(face_detector_model, max_shape, original_frames, subject_src.frame.face.frames_dir())
        subject_src.frame.face.extracting_done(max_shape)
        progress_bar.update(1)

    progress_bar.close()
