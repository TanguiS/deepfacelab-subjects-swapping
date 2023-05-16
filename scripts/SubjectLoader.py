import shutil
from pathlib import Path
from typing import Tuple

from scripts.env.workspace import WorkspaceStr


class Subject:
    def __init__(self, subject_path: Path, dim: int, quality: int) -> None:
        self.__root = subject_path
        self.__dim = dim
        self.__quality = quality
        self.__tag = subject_path.joinpath(f".tag_{dim}_{quality}")
        self.__merged = ".done"

    def id(self) -> int:
        return int(self.__root.as_posix().split("_")[-1])

    def specs(self) -> Tuple[int, int]:
        return self.__dim, self.__quality

    def video(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.videos.value)

    def swap_videos(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.s_videos.value)

    def swap_frames(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.s_frames.value)

    def merged_from(self, subject_id: int):
        return self.__root.joinpath(WorkspaceStr.s_frames.value).joinpath(WorkspaceStr.dst_video.value +
                                                                          str(subject_id))

    def frames(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.frames.value)

    def mask_from(self, subject_id: int):
        return self.__root.joinpath(WorkspaceStr.s_frames.value). \
            joinpath(WorkspaceStr.dst_video.value + str(subject_id)). \
            joinpath(WorkspaceStr.mask.value)

    def aligned(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.aligned.value)

    def root(self) -> Path:
        return self.__root

    def extract_done(self) -> None:
        self.__tag.touch(exist_ok=True)

    def is_extract_done(self):
        return self.__tag.exists()

    def merged_from_done(self, subject_id: int) -> None:
        self.merged_from(subject_id).joinpath(self.__merged).touch(exist_ok=True)

    def is_merged_from_is_done(self, subject_id: int) -> bool:
        return self.merged_from(subject_id).joinpath(self.__merged).exists()

    def clean_alignment(self) -> None:
        if self.__root.joinpath(WorkspaceStr.aligned.value).exists():
            shutil.rmtree(self.__root.joinpath(WorkspaceStr.aligned.value))
        self.__root.joinpath(WorkspaceStr.aligned.value).mkdir(exist_ok=True)
        tags = self.__root.glob(".tag*")
        for tag in tags:
            tag.unlink()

    def clean(self):
        if self.__root.joinpath(WorkspaceStr.s_videos.value).exists():
            shutil.rmtree(self.__root.joinpath(WorkspaceStr.s_videos.value))
        if self.__root.joinpath(WorkspaceStr.s_frames.value).exists():
            shutil.rmtree(self.__root.joinpath(WorkspaceStr.s_frames.value))
        if self.__root.joinpath(WorkspaceStr.frames.value).exists():
            shutil.rmtree(self.__root.joinpath(WorkspaceStr.frames.value))
        tags = self.__root.glob(".tag*")
        for tag in tags:
            tag.unlink()
