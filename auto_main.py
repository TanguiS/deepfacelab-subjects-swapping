from pathlib import Path
from typing import Union, Optional, Dict

import scripts.train.util
import scripts.util
from scripts import args_parser
from scripts.benchmark import face_swap_benchmark
from scripts.extract import facet_pack, facet_unpack, random_data_augmentation
from scripts.extract.aligned import proxy_extract
from scripts.extract.face import face_extract
from scripts.train import proxy_train, face_swap
from scripts.workspace import workspace, dataframe
from scripts.workspace.WorkspaceEnum import WorkspaceStr


def videos_to_subjects(videos_dir: Path, subjects_dir: Path) -> None:
    workspace.videos_to_subject(videos_dir, subjects_dir)


def update_workspace(subjects_dir: Path):
    subjects = workspace.load_subjects(subjects_dir)
    workspace.update_subjects(subjects)


def clean_workspace(subjects_dir: Path, redo_merged: bool, redo_original: bool, redo_face: bool) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    if redo_merged:
        for subject in subjects:
            subject.clean.clean_merged()
        dim, quality = subjects[0].specs()
        workspace.create_subject_workspace(subjects_dir, dim, quality)
        return
    if redo_original:
        for subject in subjects:
            subject.clean.clean_alignment()
        dim, quality = subjects[0].specs()
        workspace.create_subject_workspace(subjects_dir, dim, quality)
        return
    if redo_face:
        for subject in subjects:
            subject.clean.clean_face()
        dim, quality = subjects[0].specs()
        workspace.create_subject_workspace(subjects_dir, dim, quality)
        return
    for subject in subjects:
        subject.clean.clean_all()


def raw_extract(subjects_dir: Path, dim_output_faces: int, png_quality: int) -> None:
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


def face_swap_auto_action(
        subjects_dir: Path,
        model_dir: Path,
        model_name: str,
        iteration_goal: Union[int, any] = None
) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    face_swap.launch_auto(subjects, model_dir, model_name, iteration_goal)


def face_swap_train_flexible_action(
        subjects_dir: Path,
        model_dir: Path,
        model_name: str,
        iteration_goal: Union[int, any] = None
) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    face_swap.launch_flexible_train(subjects, model_dir, model_name, iteration_goal)


def face_swap_merge_flexible_action(
        subjects_dir: Path,
        model_dir: Path,
        model_name: str
) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    face_swap.launch_flexible_merge(subjects, model_dir, model_name)


def clean_flexible_train(subjects_dir: Path) -> None:
    workspace.clean_flexible_train_merge_workspace(subjects_dir)


def face_swap_bench(
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
    face_swap_benchmark.launch(
        subject_src,
        subject_dst,
        model_dir,
        benchmark_output_path_results,
        iteration_goal,
        delta_iteration
    )


def workspace_setup(args: Dict[str, any]):
    try:
        workspace.create_subject_workspace(args['subjects_dir'], args['dim_output_faces'], args['png_quality'])
    except KeyError:
        try:
            workspace.create_subject_workspace(args['subjects_dir'])
        except ValueError as e:
            print(e)


def dataframe_creation(
        subjects_dir: Path,
        output_pickle: Optional[Path] = None
) -> None:
    if output_pickle is None:
        output_pickle = subjects_dir.joinpath("dataframe.pkl")
    dataframe.create(subjects_dir, output_pickle)


def extract_face_from_subject(subjects_dir: Path, model_dir: Path, input_shape: int, max_shape: int) -> None:
    from scripts.extract.face.FaceDetectorResult import load_face_detection_model

    shape = (input_shape, input_shape)
    subjects = workspace.load_subjects(subjects_dir)
    face_detector = load_face_detection_model(model_dir, input_size=shape)
    face_extract.extract_face_from_subject(subjects, face_detector, max_shape)


def extract_face_from_video_data_augmentation(
        subjects_dir: Path,
        model_dir: Path,
        input_shape: int,
        max_shape: int
) -> None:
    from scripts.extract.face.FaceDetectorResult import load_face_detection_model
    shape = (input_shape, input_shape)
    face_detector = load_face_detection_model(model_dir, input_size=shape)
    random_data_augmentation.launch_video_extract_frames(subjects_dir)
    random_data_augmentation.launch_face_extract_frames(subjects_dir, face_detector, max_shape)


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
        "clean": (clean_workspace, {'subjects_dir', 'redo_merged', 'redo_original', 'redo_face'}),
        "extract": (raw_extract, {'subjects_dir', 'dim_output_faces', 'png_quality'}),
        "pack": (pack, {'subjects_dir'}),
        "unpack": (unpack, {'subjects_dir'}),
        "pretrain": (pretrain, {'subjects_dir', 'model_dir', 'model_name', 'model_dir_backup'}),
        "swap_auto": (face_swap_auto_action, {'subjects_dir', 'model_dir', 'model_name', 'iteration_goal'}),
        "swap_flexible_train": (face_swap_train_flexible_action, {
            'subjects_dir', 'model_dir', 'model_name', 'iteration_goal'
        }),
        "swap_flexible_merge": (face_swap_merge_flexible_action, {'subjects_dir', 'model_dir', 'model_name'}),
        "clean_flexible_train": (clean_flexible_train, {'model_dir'}),
        "face_swap_benchmark": (face_swap_bench, {
            'subjects_dir',
            'subject_src_id',
            'subject_dst_id',
            'model_dir',
            'benchmark_output_path_results',
            'iteration_goal',
            'delta_iteration'
        }),
        "dataframe": (dataframe_creation, {'subjects_dir', 'output_pickle'}),
        "extract_face_from_subject": (extract_face_from_subject, {
            'subjects_dir',
            'model_dir',
            'input_shape',
            'max_shape'
        }),
        "extract_face_from_video_data_augmentation": (extract_face_from_video_data_augmentation, {
            'subjects_dir',
            'model_dir',
            'input_shape',
            'max_shape'
        })
    }

    action = args["action"]
    workspace_setup(args)
    try:
        action_fn, action_args = actions[action]
    except KeyError:
        raise NotImplementedError(f"Action : '{action}' is not handled.")

    kwargs = {arg: args[arg] for arg in action_args}

    action_fn(**kwargs)
