import shutil
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from scripts.Subject import Subject
from scripts.workspace.WorkspaceEnum import WorkspaceStr


def videos_to_subject(input_videos_dir: Path, output_subjects_dir: Path) -> None:
    if not input_videos_dir.is_dir() or not input_videos_dir.exists() \
            or not output_subjects_dir.is_dir() or not output_subjects_dir.exists():
        raise NotADirectoryError(f"Error with input: 1: {input_videos_dir}, 2: {output_subjects_dir}")

    video_extensions = {
        ".3g2", ".3gp", ".amv", ".asf", ".avi", ".drc", ".flv", ".f4v", ".f4p", ".f4a",
        ".f4b", ".gif", ".gifv", ".m4v", ".mkv", ".mng", ".mov", ".mp2", ".mp4", ".m4p",
        ".m4v", ".mpg", ".mp2", ".mpeg", ".mpe", ".mpv", ".mpg", ".mpeg", ".m2v", ".m4v",
        ".mxf", ".nsv", ".ogg", ".ogv", ".qt", ".rm", ".rmvb", ".roq", ".svi", ".vob",
        ".webm", ".wmv", ".yuv"
    }
    videos = [
        file for file in input_videos_dir.rglob("*")
        if file.suffix.lower() in video_extensions and not file.name.startswith(".")
    ]
    next_id = len([item for item in output_subjects_dir.glob(WorkspaceStr.subject.value + "*")]) + 1

    total = len(videos)
    for video in tqdm(videos, total=total):
        subject_dir = output_subjects_dir.joinpath(WorkspaceStr.subject.value + str(next_id))
        subject_dir.mkdir(exist_ok=False)
        output_video_dir = subject_dir.joinpath(WorkspaceStr.videos.value[:-2] + video.suffix)
        shutil.copy(video, output_video_dir)
        next_id += 1


def create_subject_workspace(subjects_path: Path, dim: int, quality: int) -> None:
    subjects = load_subjects(subjects_path, dim, quality)

    for subject_1 in subjects:
        subject_1.original_frames().mkdir(exist_ok=True)
        subject_1.aligned_frames().mkdir(exist_ok=True)
        subject_1.merged_frames().mkdir(exist_ok=True)
        subject_1.merged_videos_dir().mkdir(exist_ok=True)
        for subject_2 in subjects:
            if subject_1 == subject_2:
                continue
            subject_1.merged_frames_from(subject_2.id()).mkdir(exist_ok=True)
            subject_1.mask_frames_from(subject_2.id()).mkdir(exist_ok=True)


def clean_subjects_workspace(subjects: List[Subject]) -> None:
    for subject in subjects:
        subject.clean()


def load_subjects(subjects_path: Path, dim: Optional[int] = None, quality: Optional[int] = None) -> List[Subject]:
    subjects = []
    subjects_path_list = [item for item in subjects_path.glob(WorkspaceStr.subject.value + "*")]
    for curr in subjects_path_list:
        subjects.append(Subject(curr, dim, quality))
    return subjects


def update_subjects(subjects: List[Subject]) -> None:
    max_id = len(subjects)
    remaining_id = {subject.id() for subject in subjects}
    for subject in subjects:
        if subject.id() <= max_id:
            continue
        index = 1
        for i in range(1, max_id + 1):
            if i in remaining_id:
                continue
            index = i
            remaining_id.add(index)
            break
        dst = str(subject.root_dir().parent.joinpath(WorkspaceStr.subject.value + str(index)))
        print(f"moving: {subject.root_dir()} to {dst}")
        shutil.move(str(subject.root_dir()), dst)


def benchmark_workspace(benchmark_dir: Path, models_name: List[str], max_iteration: int, delta_iteration: int) -> None:
    if not benchmark_dir.parent.exists():
        raise NotADirectoryError(f"Benchmark parent dir does not exist: {benchmark_dir.parent}")
    try:
        benchmark_dir.mkdir(exist_ok=False)
    except FileExistsError:
        if len(list(benchmark_dir.iterdir())) > 0:
            print("- Benchmark output directory is not empty!\n"
                  "  Its content will be removed, save it then press any key.")
            input(" > waiting for any key to be pressed....")
            print("-> Continuing...")
        shutil.rmtree(benchmark_dir)
        benchmark_dir.mkdir()
        benchmark_dir.joinpath(WorkspaceStr.benchmark_csv.value).touch()

    for model_name in models_name:
        model_dir = benchmark_dir.joinpath(model_name)
        model_dir.mkdir()
        model_dir.joinpath(WorkspaceStr.tmp_save.value).mkdir()
        for iteration_goal in range(delta_iteration, max_iteration + 1, delta_iteration):
            model_dir.joinpath(str(iteration_goal)).mkdir()
