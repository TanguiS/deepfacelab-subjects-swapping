import multiprocessing
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

from tqdm import tqdm

from scripts.Subject import Subject
from scripts.train import proxy_train, util
from scripts.train.util import proxy_merge_mp4
from scripts.workspace import workspace
from scripts.workspace.WorkspaceEnum import WorkspaceStr


def face_swap_train(
        subject_src: Subject,
        subject_dst: Subject,
        model_dir: Path,
        model_name: str,
        gpu_indexes: Union[list, List[int]],
        silent_start: bool = False,
        iteration_goal: Union[int, any] = None,
) -> None:
    proxy_train.launch(subject_src, subject_dst, model_dir, model_name, gpu_indexes, silent_start, iteration_goal)


def setup(model_dir: Path, model_name: str, gpu_indexes: Union[list, List[int]], subject: Subject) -> None:
    dim, quality = subject.specs()
    command = [
        "python", "auto_main.py", "pretrain",
        "--model_dir", str(model_dir), "--model_name", model_name, "--subjects_dir", str(subject.root_dir().parent)
    ]
    command_str = " ".join(command)
    print(command_str)
    while True:
        subprocess.Popen(
            f"gnome-terminal -- bash -ic 'conda activate deepfacelab; {command_str}; exec $SHELL'",
            shell=True
        )
        print("Press [Enter] when setup is done or [r] to retry and run setup again...")
        user_input = input().lower()
        if user_input == 'r':
            continue
        else:
            break
    print("Continuing...")


def setup_temporary_model(model_dir, model_name):
    files = [file for file in model_dir.glob(f"{model_name}*")]
    save_dir = model_dir.joinpath(WorkspaceStr.tmp_save.value)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    for file in tqdm(files, total=len(files), miniters=1, desc="copying model for face swapping switch"):
        if file.is_dir():
            continue
        shutil.copy(file, save_dir)


def merge(
        subject_src: Subject,
        subject_dst: Subject,
        model_dir: Path,
        model_name: str,
        gpu_indexes: Union[list, List[int], any]
) -> None:
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


def move_model(model_dir: Path, model_name: str) -> None:
    files = [file for file in model_dir.glob(f"{model_name}*")]
    for file in files:
        if file.is_dir():
            continue
        file.unlink()

    files = [file for file in model_dir.joinpath(WorkspaceStr.tmp_save.value).glob(f"{model_name}*")]
    for file in tqdm(files, total=len(files), miniters=1, desc="moving model"):
        if file.is_dir():
            continue
        shutil.copy(file, model_dir)


def launch_auto(
        subjects: List[Subject],
        model_dir: Path,
        model_name: str,
        iteration_goal: Union[int, any] = None
) -> None:
    model_name = util.choose_model(model_dir, model_name)
    gpu_indexes = util.choose_gpu_index()

    if iteration_goal is None:
        setup(model_dir, model_name, gpu_indexes, subjects[0])
    setup_temporary_model(model_dir, model_name)

    for i, subject_src in enumerate(subjects):
        for j, subject_dst in enumerate(subjects):
            if i == j or subject_src.is_merged_from_is_done(subject_dst.id()):
                subject_src.clean_mask_from(subject_dst.id())
                continue
            silent_start = i != 0
            move_model(model_dir, model_name)

            process = multiprocessing.Process(target=face_swap_train, args=(
                subject_src, subject_dst, model_dir, model_name, gpu_indexes, silent_start, iteration_goal
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
            subject_src.clean_mask_from(subject_dst.id())

    to_delete = model_dir.joinpath(WorkspaceStr.tmp_save.value)
    shutil.rmtree(to_delete)


def save_model(model_dir: Path, model_name: str, subject_src_id: int, subject_dst_id: int) -> Path:
    files = [file for file in model_dir.glob(f"{model_name}*")]
    dst_dir = model_dir.joinpath(
        WorkspaceStr.flex_model.value
    ).joinpath(f"{WorkspaceStr.model_on_sub.value}{str(subject_src_id)}_{str(subject_dst_id)}")

    for file in tqdm(files, total=len(files), miniters=1, desc="saving model"):
        if file.is_dir():
            continue
        shutil.copy(file, dst_dir)
    return dst_dir


def is_model_on_src_dst_in_retrain_mode(model_dir: Path, subject_src_id: int, subject_dst_id: int) -> bool:
    tag = model_dir.joinpath(WorkspaceStr.flex_model.value
                             ).joinpath(
        f"{WorkspaceStr.model_on_sub.value}{str(subject_src_id)}_{str(subject_dst_id)}"
    ).joinpath(WorkspaceStr.model_on_retrain_tag.value)
    is_retrain = tag.exists()
    if is_retrain:
        tag.unlink()
    return is_retrain


def is_model_on_src_dst_is_done(model_dir: Path, subject_src_id: int, subject_dst_id: int) -> bool:
    tag = model_dir.joinpath(WorkspaceStr.flex_model.value
                             ).joinpath(
        f"{WorkspaceStr.model_on_sub.value}{str(subject_src_id)}_{str(subject_dst_id)}"
    ).joinpath(WorkspaceStr.model_on_done_tag.value)
    return tag.exists()


def model_on_src_dst_is_done(model_dir: Path, subject_src_id: int, subject_dst_id: int) -> None:
    tag = model_dir.joinpath(WorkspaceStr.flex_model.value
                             ).joinpath(
        f"{WorkspaceStr.model_on_sub.value}{str(subject_src_id)}_{str(subject_dst_id)}"
    ).joinpath(WorkspaceStr.model_on_done_tag.value)
    tag.touch()


def launch_flexible_train(
        subjects: List[Subject],
        model_dir: Path,
        model_name: str,
        iteration_goal: Union[int, any] = None
) -> None:
    workspace.create_flexible_train_merge_workspace(model_dir, subjects)
    model_name = util.choose_model(model_dir, model_name)
    gpu_indexes = util.choose_gpu_index()

    if iteration_goal is None:
        setup(model_dir, model_name, gpu_indexes, subjects[0])

    for i, subject_src in enumerate(subjects):
        for j, subject_dst in enumerate(subjects):
            if i == j or is_model_on_src_dst_is_done(model_dir, subject_src.id(), subject_dst.id()):
                continue
            silent_start = i != 0
            model_on_subject = save_model(model_dir, model_name, subject_src.id(), subject_dst.id())

            process = multiprocessing.Process(target=face_swap_train, args=(
                subject_src, subject_dst, model_on_subject, model_name, gpu_indexes, silent_start, iteration_goal
            ))
            process.start()
            process.join()

    to_delete = model_dir.joinpath(WorkspaceStr.tmp_save.value)
    shutil.rmtree(to_delete)


def flexible_merge(model_dir: Path, model_name: str, subject_src: Subject, subject_dst: Subject, gpu_index: int) -> None:
    command = [
        "python", "main.py", "merge",
        "--model_class_name", 'SAEHD',
        "--saved_models_path", str(model_dir),
        "--force_model_name", model_name,
        "--input_path", str(subject_dst.original_frames()),
        "--output_path", str(subject_src.merged_frames_from(subject_dst.id())),
        "--output_mask_path", str(subject_src.mask_frames_from(subject_dst.id())),
        "--aligned_path",  str(subject_dst.aligned_frames()),
        "--forge_gpu_indexes", str(gpu_index)
    ]
    command_str = " ".join(command)
    print(command_str)
    while True:
        subprocess.Popen(
            f"gnome-terminal -- bash -ic 'conda activate deepfacelab; {command_str}; exec $SHELL'",
            shell=True
        )
        print("Press [Enter] when merge is done or [r] to retry and run merger again...")
        user_input = input().lower()
        if user_input == 'r':
            continue
        else:
            break
    print("Continuing...")


def launch_flexible_merge(
        subjects: List[Subject],
        model_dir: Path,
        model_name: str
) -> None:
    model_name = util.choose_model(model_dir, model_name)
    gpu_indexes = util.choose_gpu_index()

    for i, subject_src in enumerate(subjects):
        for j, subject_dst in enumerate(subjects):
            if i == j or subject_src.is_merged_from_is_done(subject_dst.id()):
                subject_src.clean_mask_from(subject_dst.id())
                continue
            current_model_dir = model_dir.joinpath(WorkspaceStr.flex_model.value
                                                   ).joinpath(
                f"{WorkspaceStr.model_on_sub.value}{str(subject_src.id())}_{str(subject_dst.id())}"
            )

            flexible_merge(current_model_dir, model_name, subject_src, subject_dst, gpu_indexes[0])

            subject_src.merged_done_from(subject_dst.id())
            merge_mp4(subject_src, subject_dst)
            subject_src.clean_mask_from(subject_dst.id())
