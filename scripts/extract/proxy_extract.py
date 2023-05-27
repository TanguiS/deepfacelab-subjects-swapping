from pathlib import Path
from typing import List

from core import osex
from scripts.Subject import Subject
from mainscripts import VideoEd, Extractor, Sorter, Util


def video_to_frames(input_video: Path, output_dir: Path) -> None:
    osex.set_process_lowest_prio()
    VideoEd.extract_video(
        input_file=input_video,
        output_dir=output_dir,
        output_ext='png',
        fps=0
    )


def extract_face(
    input_dir: Path,
    output_dir: Path,
    face_type: str,
    image_size: int,
    jpeg_quality: int
) -> None:
    osex.set_process_lowest_prio()
    Extractor.main(
        detector='s3fd',
        input_path=input_dir,
        output_path=output_dir,
        output_debug=False,
        manual_fix=False,
        manual_output_debug_fix=False,
        manual_window_size=1368,
        face_type=face_type,
        max_faces_from_image=1,
        image_size=image_size,
        jpeg_quality=jpeg_quality,
        cpu_only=False,
        force_gpu_idxs=[0]
    )


def sort_dir_by_hist(input_dir: Path):
    osex.set_process_lowest_prio()
    Sorter.main(
        input_path=input_dir,
        sort_by_method='hist'
    )


def recover_aligned_name(subject: Subject):
    osex.set_process_lowest_prio()
    Util.recover_original_aligned_filename(subject.aligned_frames())


def launch(subjects: List[Subject], face_type: str, image_size: int, jpeg_quality: int) -> None:
    to_recover = []
    for subject in subjects:
        if subject.is_extract_done():
            continue
        subject.clean_alignment()
        video_to_frames(subject.original_video(), subject.original_frames())
        extract_face(subject.original_frames(), subject.aligned_frames(), face_type, image_size, jpeg_quality)
        sort_dir_by_hist(subject.aligned_frames())
        subject.extract_done()
        to_recover.append(subject)

    if len(to_recover) == 0:
        return

    print("Manually Clean the Aligned face-set, then press [Enter] to continue...")
    input()
    print("Continuing...")

    for subject in to_recover:
        recover_aligned_name(subject)
