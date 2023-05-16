from pathlib import Path
from typing import List, Union

from scripts.SubjectLoader import Subject
from scripts.env.workspace import WorkspaceStr


def choose_gpu_index() -> Union[list, List[int]]:
    from core.leras import nn
    return nn.ask_choose_device_idxs(
        choose_only_one=False,
        allow_cpu=True,
        suggest_best_multi_gpu=True,
        suggest_all_gpu=False
    )


def find_models(models_dir: Path) -> List[str]:
    import os
    import operator
    from core import pathex

    saved_models_names = []
    for filepath in pathex.get_file_paths(models_dir):
        filepath_name = filepath.name
        if filepath_name.endswith(f'SAEHD_data.dat'):
            saved_models_names += [(filepath_name.split('_')[0], os.path.getmtime(filepath))]

    saved_models_names = sorted(saved_models_names, key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in saved_models_names]


def choose_model(model_dir_src: Path, model_name: str) -> str:
    from core.interact import interact as io

    if model_name == "":
        files = find_models(model_dir_src)
        model_name = io.input_str("Choose a model : ", files[0], files,
                                  help_message="Model that will be used for training")
    return model_name


def save_model_copy(model_name: str, model_dir_src: Path, model_dir_dst: Path) -> None:
    import shutil

    model_name = choose_model(model_dir_src, model_name)
    files = [file for file in model_dir_src.glob(f"{model_name}*")]
    backup_dir = model_dir_dst.joinpath(model_name)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir()
    for file in files:
        shutil.copy(file, backup_dir)


def launch(
        subject_src: Subject,
        subject_dst: Subject,
        model_dir: Path,
        model_name: str,
        gpu_indexes: any,
        silent_start: bool = False) -> None:
    from core import osex

    osex.set_process_lowest_prio()
    kwargs = {
        'model_class_name': 'SAEHD',
        'saved_models_path': model_dir,
        'training_data_src_path': subject_src.aligned(),
        'training_data_dst_path': subject_dst.aligned(),
        'pretraining_data_path': Path(WorkspaceStr.pretrain.value),
        'pretrained_model_path': None,
        'no_preview': False,
        'force_model_name': model_name if not "" else None,
        'force_gpu_idxs': gpu_indexes,
        'cpu_only': None,
        'silent_start': silent_start,
        'execute_programs': [[int(x[0]), x[1]] for x in []],
        'debug': False,
    }
    from mainscripts import Trainer
    Trainer.main(**kwargs)
