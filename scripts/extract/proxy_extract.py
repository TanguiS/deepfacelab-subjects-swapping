import shutil
from pathlib import Path
from typing import List

from core import osex
from scripts.SubjectLoader import Subject
from scripts.env.workspace import WorkspaceStr


def cut_video(input_video: Path):
    osex.set_process_lowest_prio()
    from mainscripts import VideoEd
    input_video = next(input_video.parent.glob(WorkspaceStr.videos.value))
    VideoEd.cut_video(
        input_file=input_video,
        from_time="00:00:00.000",
        to_time="00:00:10.000",
        audio_track_id=0,
        bitrate=16
    )
    output = input_video.with_name(input_video.stem+"_cut"+input_video.suffix)
    if output.exists():
        input_video.unlink()
        output.rename(output.with_name(input_video.name))


def video_to_frames(input_video: Path, output_dir: Path) -> None:
    osex.set_process_lowest_prio()
    from mainscripts import VideoEd
    VideoEd.extract_video(
        input_file=input_video,
        output_dir=output_dir,
        output_ext='png',
        fps=0
    )


def extract_face(input_dir: Path, output_dir: Path, face_type: str, image_size: int, jpeg_quality: int):
    osex.set_process_lowest_prio()
    from mainscripts import Extractor
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
    from mainscripts import Sorter
    Sorter.main(
        input_path=input_dir,
        sort_by_method='hist'
    )


def recover_aligned_name(subject: Subject):
    osex.set_process_lowest_prio()
    from mainscripts import Util
    Util.recover_original_aligned_filename(subject.aligned())


def launch(subjects: List[Subject], face_type: str, image_size: int, jpeg_quality: int) -> None:
    for subject in subjects:
        if subject.is_extract_done():
            continue
        else:
            subject.clean_alignment()
        cut_video(subject.video())
        video_to_frames(subject.video(), subject.frames())
        extract_face(subject.frames(), subject.aligned(), face_type, image_size, jpeg_quality)
        sort_dir_by_hist(subject.frames())
        subject.extract_done()
    print("Manually Clean the Aligned face-set, then press [Enter] to continue...")
    input()
    print("Continuing...")
    for subject in subjects:
        recover_aligned_name(subject)
