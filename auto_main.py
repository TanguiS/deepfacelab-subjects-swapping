from pathlib import Path
from typing import Union, List

from scripts import args_parser
from scripts.env import workspace
from scripts.train import proxy_train, face_swap
from scripts.extract import proxy_extract, facet_pack, facet_unpack
from scripts.env.workspace import WorkspaceStr


def clean_workspace(subjects_dir: Path, dim_output_faces: int, png_quality: int) -> None:
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    workspace.clean_workspace(subjects)


def extract(subjects_dir: Path, dim_output_faces: int, png_quality: int) -> None:
    subjects = workspace.create_workspace(subjects_dir, dim_output_faces, png_quality)
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
    model_name = proxy_train.choose_model(model_dir, model_name)
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    proxy_train.launch(subjects[0], subjects[1], model_dir, model_name, None)
    proxy_train.save_model_copy(model_name, model_dir, model_dir_backup)


def face_swap_action(
        subjects_dir: Path, dim_output_faces: int, png_quality: int,
        model_dir: Path, model_name: str) -> None:
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    face_swap.launch(subjects, model_dir, model_name)


def merge(
        subjects_dir: Path, dim_output_faces: int, png_quality: int,
        subject_id_src: int, subject_id_dst: int, model_dir: Path,
        model_name: str, gpu_indexes: Union[list, List[int]]) -> None:
    subjects = workspace.load_subjects(subjects_dir, dim_output_faces, png_quality)
    subject_src = None
    subject_dst = None
    for subject in subjects:
        if subject.id() == subject_id_src:
            subject_src = subject
        if subject.id() == subject_id_dst:
            subject_dst = subject
    face_swap.merge(
        subject_src=subject_src,
        subject_dst=subject_dst,
        model_dir=model_dir,
        model_name=model_name,
        gpu_indexes=None
    )


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    from core.leras import nn

    nn.initialize_main_env()

    args = args_parser.args_parser()
    root = args["subjects_dir"]
    print("args : ", args)

    actions = {
        "clean": (clean_workspace, ('subjects_dir', 'dim_output_faces', 'png_quality')),
        "extract": (extract, ('subjects_dir', 'dim_output_faces', 'png_quality')),
        "pack": (pack, ('subjects_dir', 'dim_output_faces', 'png_quality')),
        "unpack": (unpack, 'subjects_dir'),
        "pretrain": (pretrain, (
            'subjects_dir', 'dim_output_faces', 'png_quality', 'model_dir', 'model_name', 'model_dir_backup'
        )),
        "swap": (face_swap_action, ('subjects_dir', 'dim_output_faces', 'png_quality', 'model_dir', 'model_name')),
        "merge": (merge, (
            'subjects_dir', 'dim_output_faces', 'png_quality', 'subject_id_src',
            'subject_id_dst', 'model_dir', 'model_name', 'gpu_indexes'
        ))
    }

    action = args["action"]
    try:
        action_fn, action_args = actions[action]
    except KeyError:
        raise NotImplementedError(f"Action : '{action}' is not handled.")

    kwargs = {arg: args[arg] for arg in action_args}
    print("kwargs : ", kwargs)
    action_fn(**kwargs)
