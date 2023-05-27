from pathlib import Path
import enum
from typing import List

from tqdm import tqdm


class WorkspaceStr(enum.Enum):
    frames = "frames"
    aligned = frames + "/aligned"
    s_videos = "merged_videos"
    s_frames = "merged_frames"
    subject = "subject_"
    dst_video = "from_"
    videos = "output.*"
    tag = ".tag"
    pretrain = "pretrain_faces"
    mask = "mask"
    tmp_save = "tmp_save"
    metadata = "metadata.json"
    benchmark_csv = "benchmark.csv"


from scripts.Subject import Subject


def videos_to_subject(input_videos_dir: Path, output_subjects_dir: Path) -> None:
    import shutil

    if (not input_videos_dir.is_dir() or not input_videos_dir.exists()
            or not output_subjects_dir.is_dir() or not output_subjects_dir.exists()):
        raise NotADirectoryError(f"Error with input : 1: {input_videos_dir}, 2 : {output_subjects_dir}")

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
    subjects = []
    subjects_path.joinpath(WorkspaceStr.pretrain.value).mkdir(exist_ok=True)
    subjects_path = [path for path in subjects_path.glob(WorkspaceStr.subject.value + "*")]
    for curr in subjects_path:
        for path in (curr.joinpath(WorkspaceStr.frames.value),
                     curr.joinpath(WorkspaceStr.aligned.value),
                     curr.joinpath(WorkspaceStr.s_frames.value),
                     curr.joinpath(WorkspaceStr.s_videos.value)):
            path.mkdir(exist_ok=True)
        subjects.append(Subject(curr, dim, quality))
        for i in range(1, len(subjects_path) + 1):
            if i == subjects[-1].id():
                continue
            sub = subjects[-1].merged_frames().joinpath(WorkspaceStr.dst_video.value + f"{i}")
            sub.mkdir(exist_ok=True)
            sub2 = sub.joinpath(WorkspaceStr.mask.value)
            sub2.mkdir(exist_ok=True)


def clean_subjects_workspace(subjects: List[Subject]) -> None:
    for subject in subjects:
        subject.clean()


def load_subjects(subjects_path: Path, dim: int, quality: int) -> List[Subject]:
    from tqdm import tqdm

    subjects = []
    tmp = [item for item in subjects_path.glob(WorkspaceStr.subject.value + "*")]
    for curr in tqdm(tmp, total=len(tmp), desc="loading subjects", miniters=1.0, unit="subject"):
        subjects.append(Subject(curr, dim, quality))
    return subjects


def update_subjects(subjects: List[Subject]) -> None:
    import shutil

    remaining_ids = {subject.id() for subject in subjects}
    max_id = len(subjects)

    for subject in subjects:
        if subject.id() <= max_id:
            continue
        index = 1
        for i in range(1, max_id + 1):
            if i in remaining_ids:
                continue
            index = i
            remaining_ids.add(index)
            remaining_ids.remove(subject.id())
            break
        shutil.move(str(subject.root_dir()), str(subject.root_dir().parent.joinpath(
            WorkspaceStr.subject.value + str(index))))


def benchmark_workspace(benchmark_dir: Path, models_name: List[str], max_iteration: int, delta_iteration: int) -> None:
    import shutil

    if not benchmark_dir.parent.exists():
        raise NotADirectoryError(f"Benchmark parent dir does not exists : {benchmark_dir.parent}")
    try:
        benchmark_dir.mkdir(exist_ok=False)
    except FileExistsError:
        if len([f for f in benchmark_dir.iterdir()]) > 0:
            print(
                "- Benchmark output directory is not empty ! \n" +
                "  its content will be removed, save it then press any key."
            )
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


