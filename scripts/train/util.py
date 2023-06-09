from pathlib import Path
from typing import Union, List
from core import osex
from core.leras import nn
from core.interact import interact as io
from mainscripts import VideoEd, Merger
from core import pathex
import os
import operator


def proxy_merge_mp4(input_dir: Path, output_file: Path, reference_file: Path) -> None:
    osex.set_process_lowest_prio()
    VideoEd.video_from_sequence(
        input_dir=input_dir,
        output_file=output_file,
        reference_file=reference_file,
        ext="png",
        fps=0,
        bitrate=16,
        include_audio=False,
        lossless=False
    )


def proxy_merge(
    model_dir: Path,
    model_name: str,
    input_path: Path,
    output_path: Path,
    output_mask_path: Path,
    aligned_path,
    gpu_indexes: Union[list, List[int], any]
) -> None:
    osex.set_process_lowest_prio()
    Merger.main(
        model_class_name='SAEHD',
        saved_models_path=model_dir,
        force_model_name=model_name or None,
        input_path=input_path,
        output_path=output_path,
        output_mask_path=output_mask_path,
        aligned_path=aligned_path,
        force_gpu_idxs=gpu_indexes,
        cpu_only=None
    )


def choose_gpu_index() -> Union[list, List[int]]:
    return nn.ask_choose_device_idxs(
        choose_only_one=False,
        allow_cpu=True,
        suggest_best_multi_gpu=True,
        suggest_all_gpu=False
    )


def find_models(models_dir: Path) -> List[str]:
    saved_models_names = []
    for filepath in pathex.get_file_paths(models_dir):
        filepath_name = filepath.name
        if filepath_name.endswith(f'SAEHD_data.dat'):
            saved_models_names += [(filepath_name.split('_')[0], os.path.getmtime(filepath))]

    saved_models_names = sorted(saved_models_names, key=operator.itemgetter(1), reverse=True)
    return [x[0] for x in saved_models_names]


def choose_model(model_dir_src: Path, model_name: str) -> str:
    if model_name == "":
        files = find_models(model_dir_src)
        model_name = io.input_str("Choose a model: ", default_value=files[0], valid_list=files,
                                  help_message="Model that will be used for training")
    return model_name
