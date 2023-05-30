from pathlib import Path
from typing import Union

import scripts.train.util
import scripts.util
from scripts import args_parser
from scripts.workspace import workspace
from scripts.train import proxy_train, face_swap
from scripts.extract import proxy_extract, facet_pack, facet_unpack
from scripts.workspace.WorkspaceEnum import WorkspaceStr
from scripts.benchmark import benchmark


def videos_to_subjects(videos_dir: Path, subjects_dir: Path) -> None:
    workspace.videos_to_subject(videos_dir, subjects_dir)


def update_workspace(subjects_dir: Path):
    subjects = workspace.load_subjects(subjects_dir)
    workspace.update_subjects(subjects)


def clean_workspace(subjects_dir: Path) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    workspace.clean_subjects_workspace(subjects)


def extract(subjects_dir: Path, dim_output_faces: int, png_quality: int) -> None:
    workspace.create_subject_workspace(subjects_dir, dim_output_faces, png_quality)
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    proxy_extract.launch(subjects, 'whole_face', dim_output_faces, png_quality)


def pack(subjects_dir: Path) -> None:
    dst = subjects_dir.joinpath(WorkspaceStr.pretrain.value)
    subjects = workspace.load_subjects(subjects_dir)
    facet_pack.launch(subjects, dst)


def unpack(subjects_dir: Path) -> None:
    dst = subjects_dir.joinpath(WorkspaceStr.pretrain.value)
    facet_unpack.launch(dst)


def pretrain(subjects_dir: Path, model_dir: Path, model_name: str, model_dir_backup: Path) -> None:
    model_name = scripts.train.util.choose_model(model_dir, model_name)
    subjects = workspace.load_subjects(subjects_dir)
    proxy_train.launch(subjects[0], subjects[1], model_dir, model_name, None)
    if model_dir_backup is not None and model_dir_backup.exists() and model_dir_backup.is_dir():
        proxy_train.save_model_copy(model_name, model_dir, model_dir_backup)


def face_swap_action(
        subjects_dir: Path,
        model_dir: Path,
        model_name: str,
        iteration_goal: Union[int, any] = None
) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    face_swap.launch(subjects, model_dir, model_name, iteration_goal)


def bench(
        subjects_dir: Path,
        subject_src_id: int,
        subject_dst_id: int,
        model_dir: Path,
        benchmark_output_path_results: Path,
        iteration_goal: int,
        delta_iteration: int
) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    subject_src, subject_dst = None, None
    for subject in subjects:
        if subject.id() == subject_src_id:
            subject_src = subject
        if subject.id() == subject_dst_id:
            subject_dst = subject
    if subject_src is None or subject_dst is None:
        raise IndexError(f"Wrong given indexes from src: {subject_src_id} or dst: {subject_dst_id}")
    benchmark.launch(
        subject_src,
        subject_dst,
        model_dir,
        benchmark_output_path_results,
        iteration_goal,
        delta_iteration
    )


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    from core.leras import nn

    nn.initialize_main_env()

    args = args_parser.args_parser()
    print("args : ", args)

    actions = {
        "to_subject": (videos_to_subjects, {'videos_dir', 'subjects_dir'}),
        "update_wrk": (update_workspace, {'subjects_dir'}),
        "clean": (clean_workspace, {'subjects_dir'}),
        "extract": (extract, {'subjects_dir', 'dim_output_faces', 'png_quality'}),
        "pack": (pack, {'subjects_dir'}),
        "unpack": (unpack, {'subjects_dir'}),
        "pretrain": (pretrain, {'subjects_dir', 'model_dir', 'model_name', 'model_dir_backup'}),
        "swap": (face_swap_action, {'subjects_dir', 'model_dir', 'model_name', 'iteration_goal'}),
        "benchmark": (bench, {
            'subjects_dir',
            'subject_src_id',
            'subject_dst_id',
            'model_dir',
            'benchmark_output_path_results',
            'iteration_goal',
            'delta_iteration'
        })
    }

    action = args["action"]
    try:
        action_fn, action_args = actions[action]
    except KeyError:
        raise NotImplementedError(f"Action : '{action}' is not handled.")

    kwargs = {arg: args[arg] for arg in action_args}

    action_fn(**kwargs)
