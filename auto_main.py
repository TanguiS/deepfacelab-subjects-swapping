from pathlib import Path

import scripts.train.util
import scripts.util
from scripts import args_parser
from scripts.workspace import workspace
from scripts.train import proxy_train, face_swap
from scripts.extract import proxy_extract, facet_pack, facet_unpack
from scripts.workspace.workspace import WorkspaceStr


def videos_to_subjects(videos_dir: Path, subjects_dir: Path) -> None:
    workspace.videos_to_subject(videos_dir, subjects_dir)


def update_workspace(subjects_dir: Path, dim_output_faces: int, png_quality: int):
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    workspace.update_subjects(subjects)


def clean_workspace(subjects_dir: Path, dim_output_faces: int, png_quality: int) -> None:
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    workspace.clean_subjects_workspace(subjects)


def extract(subjects_dir: Path, dim_output_faces: int, png_quality: int) -> None:
    workspace.create_subject_workspace(subjects_dir, dim_output_faces, png_quality)
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    proxy_extract.launch(subjects, 'whole_face', dim_output_faces, png_quality)


def pack(subjects_dir: Path, dim_output_faces: int, png_quality: int) -> None:
    dst = subjects_dir.joinpath(WorkspaceStr.pretrain.value)
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    facet_pack.launch(subjects, dst)


def unpack(subjects_dir: Path) -> None:
    dst = subjects_dir.joinpath(WorkspaceStr.pretrain.value)
    facet_unpack.launch(dst)


def pretrain(
        subjects_dir: Path, dim_output_faces: int, png_quality: int,
        model_dir: Path, model_name: str, model_dir_backup: Path) -> None:
    model_name = scripts.train.util.choose_model(model_dir, model_name)
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    proxy_train.launch(subjects[0], subjects[1], model_dir, model_name, None)
    if model_dir_backup is not None and model_dir_backup.exists() and model_dir_backup.is_dir():
        proxy_train.save_model_copy(model_name, model_dir, model_dir_backup)


def face_swap_action(
        subjects_dir: Path, dim_output_faces: int, png_quality: int,
        model_dir: Path, model_name: str) -> None:
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    face_swap.launch(subjects, model_dir, model_name)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    from core.leras import nn

    nn.initialize_main_env()

    args = args_parser.args_parser()
    root = args["subjects_dir"]
    print("args : ", args)

    actions = {
        "to_subject": (videos_to_subjects, ('videos_dir', 'subjects_dir')),
        "update_wrk": (update_workspace, ('subjects_dir', 'dim_output_faces', 'png_quality')),
        "clean": (clean_workspace, ('subjects_dir', 'dim_output_faces', 'png_quality')),
        "extract": (extract, ('subjects_dir', 'dim_output_faces', 'png_quality')),
        "pack": (pack, ('subjects_dir', 'dim_output_faces', 'png_quality')),
        "unpack": (unpack, 'subjects_dir'),
        "pretrain": (pretrain, (
            'subjects_dir', 'dim_output_faces', 'png_quality', 'model_dir', 'model_name', 'model_dir_backup'
        )),
        "swap": (face_swap_action, ('subjects_dir', 'dim_output_faces', 'png_quality', 'model_dir', 'model_name'))
    }

    action = args["action"]
    try:
        action_fn, action_args = actions[action]
    except KeyError:
        raise NotImplementedError(f"Action : '{action}' is not handled.")

    kwargs = {arg: args[arg] for arg in action_args}
    action_fn(**kwargs)
