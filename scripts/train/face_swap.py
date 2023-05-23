import multiprocessing
import subprocess
from pathlib import Path
from typing import List, Union

from scripts.Subject import Subject
from scripts.workspace.workspace import WorkspaceStr
from scripts.train import proxy_train


def merge_to_mp4(subject_src: Subject, subject_dst: Subject) -> None:
    from core import osex

    osex.set_process_lowest_prio()
    from mainscripts import VideoEd
    VideoEd.video_from_sequence(
        input_dir=subject_src.merged_frames_from(subject_dst.id()),
        output_file=subject_src.merged_videos_dir().joinpath(f"result_from_{subject_dst.id()}.mp4"),
        reference_file=subject_dst.original_video(),
        ext="png",
        fps=0,
        bitrate=16,
        include_audio=False,
        lossless=False
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
        input_path=subject_dst.original_frames(),
        output_path=subject_src.merged_frames_from(subject_dst.id()),
        output_mask_path=subject_src.mask_frames_from(subject_dst.id()),
        aligned_path=subject_dst.aligned_frames(),
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


def setup(model_dir: Path, model_name: str, gpu_indexes: Union[list, List[int]], subject: Subject) -> None:
    import shutil

    dim, quality = subject.specs()
    command = [
        "python", "auto_main.py", "--action", "pretrain",
        "--dim_output_faces", str(dim), "--png_quality", str(quality),
        "--model_dir", str(model_dir), "--model_name", model_name,
        "--model_dir_backup", str(Path("none"))
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

    files = [file for file in model_dir.glob(f"{model_name}*")]
    save_dir = model_dir.joinpath(WorkspaceStr.tmp_save.value)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    for file in files:
        if file.is_dir():
            continue
        shutil.copy(file, save_dir)


def swap_model(model_dir: Path, model_name: str) -> None:
    import shutil

    files = [file for file in model_dir.glob(f"{model_name}*")]
    for file in files:
        if file.is_dir():
            continue
        file.unlink()

    files = [file for file in model_dir.joinpath(WorkspaceStr.tmp_save.value).glob(f"{model_name}*")]
    for file in files:
        if file.is_dir():
            continue
        shutil.copy(file, model_dir)


def launch(subjects: List[Subject], model_dir: Path, model_name: str) -> None:
    model_name = proxy_train.choose_model(model_dir, model_name)
    gpu_indexes = proxy_train.choose_gpu_index()
    dim, quality = subjects[0].specs()
    setup(model_dir, model_name, gpu_indexes, subjects[0])
    for i, subject_src in enumerate(subjects):
        for j, subject_dst in enumerate(subjects):
            if i == j:
                continue
            if subject_src.is_merged_from_is_done(subject_dst.id()):
                continue
            silent_start = True
            if i == 0:
                silent_start = False
            swap_model(model_dir, model_name)
            """
            command = [
                "python", "auto_main.py", "--action", "faces_train",
                "--subjects_dir", str(subjects[0].root().parent), "--dim_output_faces", str(dim), "--png_quality",
                str(quality), "--subject_id_src", str(subject_src.id()), "--subject_id_dst", str(subject_dst.id()),
                "--model_dir", str(model_dir), "--model_name", model_name,
                "--gpu_indexes", encode_gpu_indexes(gpu_indexes), "--silent_start", str(silent_start)
            ]
            command_str = " ".join(command)
            print(command_str)
            # process = subprocess.Popen(f"gnome-terminal -- bash -ic 'conda activate deepfacelab; {command_str}; exec $SHELL'", shell=True)

            # face_swap_train(subject_src, subject_dst, model_dir, model_name, gpu_indexes, silent_start)

            # process.wait()
            """

            process = multiprocessing.Process(target=face_swap_train, args=(
                subject_src, subject_dst, model_dir, model_name, gpu_indexes, silent_start))
            process.start()
            process.join()

            """
            command = [
                "python", "auto_main.py", "--action", "merge",
                "--subjects_dir", str(subjects[0].root().parent), "--dim_output_faces", str(dim), "--png_quality",
                str(quality), "--subject_id_src", str(subject_src.id()), "--subject_id_dst", str(subject_dst.id()),
                "--model_dir", str(model_dir), "--model_name", model_name,
                "--gpu_indexes", encode_gpu_indexes(gpu_indexes)
            ]
            command_str = " ".join(command)
            print(command_str)
            """

            process = multiprocessing.Process(target=merge, args=(
                subject_src, subject_dst, model_dir, model_name, gpu_indexes))
            process.start()
            process.join()

            """
            while True:
                subprocess.Popen(f"gnome-terminal -- bash -ic 'conda activate deepfacelab; {command_str}; exec $SHELL'",
                                 shell=True)
                print("Press [Enter] when merge is done or [r] to retry and run merger again...")
                user_input = input().lower()
                if user_input == 'r':
                    continue
                else:
                    break
            """
            subject_src.merged_done_from(subject_dst.id())
            merge_to_mp4(subject_src, subject_dst)
