import multiprocessing
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

from tqdm import tqdm

import scripts.train.util
import scripts.util
from scripts.Subject import Subject
from scripts.train.util import proxy_merge_mp4
from scripts.workspace.WorkspaceEnum import WorkspaceStr
from scripts.train import proxy_train, util


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
        "--model_dir", str(model_dir), "--model_name", model_name, "--subjects_dir", str(subject.root_dir().parent)
    ]
    command_str = " ".join(command)
    print(command_str)
    while True:
        subprocess.Popen(f"gnome-terminal -- bash -ic 'conda activate deepfacelab; {command_str}; exec $SHELL'",
                         shell=True)
        print("Press [Enter] when setup is done or [r] to retry and run setup again...")
        user_input = input().lower()
        if user_input == 'r':
            continue
        else:
            break
    print("Continuing...")

    files = [file for file in model_dir.glob(f"{model_name}*")]
    save_dir = model_dir.joinpath(WorkspaceStr.tmp_save.value)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    for file in tqdm(files, total=len(files), miniters=1, desc="copying model for face swapping switch"):
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
    for file in tqdm(files, total=len(files), miniters=1, desc="swapping model"):
        if file.is_dir():
            continue
        shutil.copy(file, model_dir)


def merge(
        subject_src: Subject,
        subject_dst: Subject,
        model_dir: Path,
        model_name: str,
        gpu_indexes: Union[list, List[int], any]) -> None:
    util.proxy_merge(
        model_dir=model_dir,
        model_name=model_name,
        input_path=subject_dst.original_frames(),
        output_path=subject_src.merged_frames_from(subject_dst.id()),
        output_mask_path=subject_src.mask_frames_from(subject_dst.id()),
        aligned_path=subject_dst.aligned_frames(),
        gpu_indexes=gpu_indexes
    )


def merge_mp4(subject_src: Subject, subject_dst: Subject) -> None:
    proxy_merge_mp4(
        input_dir=subject_src.merged_frames_from(subject_dst.id()),
        output_file=subject_src.merged_videos_dir().joinpath(f"result_from_{subject_dst.id()}.mp4"),
        reference_file=subject_dst.original_video()
    )


def launch(subjects: List[Subject], model_dir: Path, model_name: str) -> None:
    model_name = scripts.train.util.choose_model(model_dir, model_name)
    gpu_indexes = scripts.train.util.choose_gpu_index()
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

            process = multiprocessing.Process(target=face_swap_train, args=(
                subject_src, subject_dst, model_dir, model_name, gpu_indexes, silent_start
            ))
            process.start()
            process.join()

            process = multiprocessing.Process(target=merge, args=(
                subject_src, subject_dst, model_dir, model_name, gpu_indexes
            ))
            process.start()
            process.join()

            subject_src.merged_done_from(subject_dst.id())
            merge_mp4(subject_src, subject_dst)
    shutil.rmtree(WorkspaceStr.tmp_save.value)
