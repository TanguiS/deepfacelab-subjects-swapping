import subprocess
from pathlib import Path
from typing import List, Union

from scripts.SubjectLoader import Subject
from scripts.train import proxy_train


def merge_to_mp4(subject_src: Subject, subject_dst: Subject) -> None:
    from core import osex

    osex.set_process_lowest_prio()
    from mainscripts import VideoEd
    VideoEd.video_from_sequence(
        input_dir=subject_src.merged_from(subject_dst.id()),
        output_file=subject_src.swap_videos().joinpath(f"result_from_{subject_dst.id()}.mp4"),
        reference_file=subject_dst.video(),
        ext="png",
        fps=None,
        bitrate=16,
        include_audio=False,
        lossless=True
    )


def merge(
        subject_src: Subject, subject_dst: Subject, model_dir: Path,
        model_name: str, gpu_indexes: Union[list, List[int], any]) -> None:
    from core import osex

    osex.set_process_lowest_prio()
    from mainscripts import Merger
    Merger.main(
        model_class_name='SAEHD',
        saved_models_path=model_dir,
        force_model_name=model_name if not "" else None,
        input_path=subject_dst.frames(),
        output_path=subject_src.merged_from(subject_dst.id()),
        output_mask_path=subject_src.mask_from(subject_dst.id()),
        aligned_path=subject_dst.aligned(),
        force_gpu_idxs=gpu_indexes,
        cpu_only=None
    )


def face_swap_train(
        subject_src: Subject,
        subject_dst: Subject,
        model_dir: Path,
        model_name: str,
        gpu_indexes: Union[list, List[int]],
        silent_start:
        bool = False) -> None:
    proxy_train.launch(subject_src, subject_dst, model_dir, model_name, gpu_indexes, silent_start)


def encode_gpu_indexes(gpu_indexes: Union[List, List[int]]) -> str:
    str_list = [str(idx) for idx in gpu_indexes]
    result = ",".join(str_list)
    return result


def launch(subjects: List[Subject], model_dir: Path, model_name: str) -> None:
    model_name = proxy_train.choose_model(model_dir, model_name)
    gpu_indexes = proxy_train.choose_gpu_index()
    dim, quality = subjects[0].specs()
    for i, subject_src in enumerate(subjects):
        for j, subject_dst in enumerate(subjects):
            if i == j:
                continue
            if subject_src.is_merged_from_is_done(subject_dst.id()):
                continue
            if i == 0:
                face_swap_train(subject_src, subject_dst, model_dir, model_name, gpu_indexes)
            else:
                face_swap_train(subject_src, subject_dst, model_dir, model_name, gpu_indexes, silent_start=True)
            command = [
                "python", "auto_main.py", "--action", "merge",
                "--subjects_dir", str(subjects[0].root().parent), "--dim_output_faces", str(dim), "--png_quality",
                str(quality), "--subject_id_src", str(subject_src.id()), "--subject_id_dst", str(subject_dst.id()),
                "--model_dir", str(model_dir), "--model_name", model_name,
                "--gpu_indexes", encode_gpu_indexes(gpu_indexes)
            ]
            command_str = " ".join(command)
            print(command_str)

            while True:
                subprocess.Popen(f"gnome-terminal -- bash -ic 'conda activate deepfacelab; {command_str}; exec $SHELL'",
                                 shell=True)
                print("Press [Enter] when merge is done or [r] to retry and run merger again...")
                user_input = input().lower()
                if user_input == 'r':
                    continue
                else:
                    break
            subject_src.merged_from_done(subject_dst.id())
            merge_to_mp4(subject_src, subject_dst)
