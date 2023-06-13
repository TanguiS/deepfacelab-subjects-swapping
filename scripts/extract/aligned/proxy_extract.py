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
    Util.recover_original_aligned_filename(subject.frame.original.aligned_dir())


def raw_launch(subjects: List[Subject], face_type: str, image_size: int, jpeg_quality: int) -> None:
    to_recover = []
    for subject in subjects:
        if subject.frame.original.is_align_extract_done():
            continue
        subject.clean.clean_alignment()
        video_to_frames(subject.video.original_video(), subject.frame.original.frames_dir())
        extract_face(subject.frame.original.frames_dir(), subject.frame.original.aligned_dir(), face_type, image_size, jpeg_quality)
        sort_dir_by_hist(subject.frame.original.aligned_dir())
        subject.frame.original.align_extract_done()
        to_recover.append(subject)

    if len(to_recover) == 0:
        return

    print("Manually Clean the Aligned face-set, then press [Enter] to continue...")
    input()
    print("Continuing...")

    for subject in to_recover:
        recover_aligned_name(subject)


"""
def swap_launch(subjects: List[Subject], face_type: str, image_size: int, jpeg_quality: int) -> None:
    for subject_src in subjects:
        for subject_dst in subjects:
            if subject_src == subject_dst:
                continue
            if subject_src.is_face_extract_done_from(subject_dst.id()):
                continue
            extract_face(
                subject_src.merged_frames_from(subject_dst.id()),
                subject_src.face_merged_frames_from(subject_dst.id()),
                face_type,
                image_size,
                jpeg_quality
            )
            subject_src.face_extract_done_from(subject_dst.id())
"""
