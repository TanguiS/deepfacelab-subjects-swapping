from pathlib import Path
import enum
from typing import List


class WorkspaceStr(enum.Enum):
    frames = "frames"
    aligned = frames + "/aligned"
    s_videos = "swapping_videos"
    s_frames = "swapping_frames"
    subject = "subject_"
    dst_video = "from_"
    videos = "output.*"
    tag = ".tag"
    pretrain = "pretrain_faces"
    mask = "mask"


from scripts.SubjectLoader import Subject


def create_workspace(subjects_path: Path, dim: int, quality: int) -> None:
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
            sub = subjects[-1].swap_frames().joinpath(WorkspaceStr.dst_video.value + f"{i}")
            sub.mkdir(exist_ok=True)
            sub2 = sub.joinpath(WorkspaceStr.mask.value)
            sub2.mkdir(exist_ok=True)


def clean_workspace(subjects: List[Subject]) -> None:
    for subject in subjects:
        subject.clean()


def load_subjects(subjects_path: Path, dim: int, quality: int) -> List[Subject]:
    from tqdm import tqdm

    subjects = []
    total = len([item for item in subjects_path.glob(WorkspaceStr.subject.value + "*")])
    for curr in tqdm(subjects_path.glob(WorkspaceStr.subject.value + "*"), total=total, desc="loading subjects", miniters=1.0, unit="subjects/s"):
        subjects.append(Subject(curr, dim, quality))
    return subjects
