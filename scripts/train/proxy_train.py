from pathlib import Path
from typing import Union

from tqdm import tqdm

from scripts.Subject import Subject
from scripts.train.util import choose_model
from scripts.workspace.workspace import WorkspaceStr


def save_model_copy(model_name: str, model_dir_src: Path, model_dir_dst: Path) -> None:
    import shutil

    model_name = choose_model(model_dir_src, model_name)
    files = [file for file in model_dir_src.glob(f"{model_name}*")]
    backup_dir = model_dir_dst.joinpath(model_name)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir()
    for file in tqdm(files, total=len(files), desc="Copying pretrained model"):
        if file.is_dir():
            continue
        shutil.copy(file, backup_dir)


def launch(
        subject_src: Subject,
        subject_dst: Subject,
        model_dir: Path,
        model_name: str,
        gpu_indexes: any,
        silent_start: bool = False,
        target_iter_args: Union[int, any] = None) -> None:
    from core import osex

    osex.set_process_lowest_prio()
    kwargs = {
        'model_class_name': 'SAEHD',
        'saved_models_path': model_dir,
        'training_data_src_path': subject_src.aligned_frames(),
        'training_data_dst_path': subject_dst.aligned_frames(),
        'pretraining_data_path': subject_src.root_dir().parent.joinpath(WorkspaceStr.pretrain.value),
        'pretrained_model_path': None,
        'no_preview': False,
        'force_model_name': model_name if not "" else None,
        'force_gpu_idxs': gpu_indexes,
        'cpu_only': None,
        'silent_start': silent_start,
        'execute_programs': [[int(x[0]), x[1]] for x in []],
        'debug': False,
        'target_iter_args': target_iter_args
    }
    from mainscripts import Trainer
    Trainer.main(**kwargs)
