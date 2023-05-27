import csv
import multiprocessing
import shutil
from datetime import datetime
from pathlib import Path

from scripts.Subject import Subject
from scripts.train import util, proxy_train
from scripts.workspace import workspace
from scripts.workspace.WorkspaceEnum import WorkspaceStr


def write_bench(output_dir: Path, model_name: str, target_iteration: int, start_time: str, end_time: str):
    with open(output_dir.joinpath(WorkspaceStr.benchmark_csv.value), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model_name, str(target_iteration), start_time, end_time])


def run_proxy_train(subject_src: Subject, subject_dst: Subject, models_dir: Path, model_name: str,
                    gpu_indexes: list, iteration_goal: int):
    proxy_train.launch(subject_src, subject_dst, models_dir, model_name, gpu_indexes, False, iteration_goal)


def run_proxy_merge(models_dir: Path, model_name: str, input_dir: Path, output_dir: Path,
                    mask_dir: Path, aligned_dir: Path, gpu_indexes: list):
    util.proxy_merge(models_dir, model_name, input_dir, output_dir, mask_dir, aligned_dir, gpu_indexes)


def run_merge_mp4(output_dir: Path, current_iteration: int, subject_dst: Subject):
    current_output_dir = output_dir.joinpath(str(current_iteration))
    util.proxy_merge_mp4(input_dir=current_output_dir,
                         output_file=current_output_dir.joinpath("output.mp4"),
                         reference_file=subject_dst.original_video())


def launch(
        subject_src: Subject, subject_dst: Subject, models_dir: Path,
        output_dir: Path, max_iteration: int, delta_iteration: int) -> None:
    models_name = util.find_models(models_dir)
    gpu_indexes = util.choose_gpu_index()
    workspace.benchmark_workspace(output_dir, models_name, max_iteration, delta_iteration)

    for model_name in models_name:
        tmp_mask_dir = output_dir.joinpath(model_name).joinpath(WorkspaceStr.tmp_save.value)

        for iteration_goal in range(delta_iteration, max_iteration + 1, delta_iteration):
            current_output_dir = output_dir.joinpath(model_name).joinpath(str(iteration_goal))

            process_train = multiprocessing.Process(target=run_proxy_train, args=(
                subject_src, subject_dst, models_dir, model_name, gpu_indexes, iteration_goal
            ))
            start_time = datetime.now().strftime("%H:%M:%S")
            process_train.start()
            process_train.join()
            end_time = datetime.now().strftime("%H:%M:%S")

            process_merge = multiprocessing.Process(target=run_proxy_merge, args=(
                models_dir, model_name, subject_dst.original_frames(), current_output_dir, tmp_mask_dir,
                subject_dst.aligned_frames(), gpu_indexes
            ))
            process_merge.start()
            process_merge.join()

            run_merge_mp4(output_dir, iteration_goal, subject_dst)

            write_bench(output_dir, model_name, iteration_goal, start_time, end_time)

        shutil.rmtree(tmp_mask_dir)
