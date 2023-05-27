import shutil
from pathlib import Path
from typing import Tuple, Union

from scripts.workspace.WorkspaceEnum import WorkspaceStr


class Subject:
    def __init__(self, subject_path: Path, dim: Union[int, any] = None, quality: Union[int, any] = None) -> None:
        self.__root = subject_path
        if dim is None or quality is None:
            raise NotImplementedError("feature not implemented yet, please provide dim and quality.")
        self.__dim = dim
        self.__quality = quality
        self.__tag = subject_path.joinpath(f".tag_{dim}_{quality}")
        self.__merged = ".done"

    def id(self) -> int:
        return int(self.__root.as_posix().split("_")[-1])

    def specs(self) -> Tuple[int, int]:
        return self.__dim, self.__quality

    def root_dir(self) -> Path:
        return self.__root

    def original_video(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.videos.value)

    def original_frames(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.frames.value)

    def metadata(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.metadata.value)

    def reset_metadata(self) -> None:
        if self.metadata().exists():
            self.metadata().unlink()
        self.metadata().touch()

    def merged_videos_dir(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.s_videos.value)

    def merged_videos_from(self, subject_id: int) -> Path:
        return self.merged_frames().joinpath(f"result_from_{subject_id}.mp4")

    def merged_frames(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.s_frames.value)

    def merged_frames_from(self, subject_id: int):
        return self.__root.joinpath(WorkspaceStr.s_frames.value).joinpath(WorkspaceStr.dst_video.value +
                                                                          str(subject_id))

    def mask_frames_from(self, subject_id: int):
        return self.__root.joinpath(WorkspaceStr.s_frames.value). \
            joinpath(WorkspaceStr.dst_video.value + str(subject_id)). \
            joinpath(WorkspaceStr.mask.value)

    def aligned_frames(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.aligned.value)

    def extract_done(self) -> None:
        self.__tag.touch(exist_ok=True)

    def is_extract_done(self):
        return self.__tag.exists()

    def merged_done_from(self, subject_id: int) -> None:
        self.merged_frames_from(subject_id).joinpath(self.__merged).touch(exist_ok=True)

    def is_merged_from_is_done(self, subject_id: int) -> bool:
        return self.merged_frames_from(subject_id).joinpath(self.__merged).exists()

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
        if self.metadata().exists():
            self.metadata().unlink()
