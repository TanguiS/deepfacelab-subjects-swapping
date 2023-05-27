from pathlib import Path
from typing import Tuple, Union, List

from scripts.workspace.WorkspaceEnum import WorkspaceStr


class Subject:
    def __init__(self, subject_path: Path, dim: Union[int, any] = None, quality: Union[int, any] = None) -> None:
        self.__root = subject_path

        if dim is None or quality is None:
            available_tags = self.__find_available_tags()
            if len(available_tags) == 0:
                raise ValueError("No available tags found in the subject folder.")

            elif len(available_tags) == 1:
                print(f"Found 1 tag : {available_tags[0]}")
                self.__dim, self.__quality = self.__parse_subject_tags(available_tags[0])
            else:
                print("/!\\ More than 1 tag is present, clean the workspace.")
                raise Exception("Feature not implemented.")
        else:
            self.__dim = dim
            self.__quality = quality

        self.__tag = subject_path.joinpath(f".tag_{self.__dim}_{self.__quality}")
        self.__merged = ".done"

    def __find_available_tags(self) -> List[Path]:
        tags = list(self.__root.glob(".tag_*"))
        return tags

    def __parse_subject_tags(self, tag: Path) -> Tuple[int, int]:
        tag_name = tag.name[5:]
        dim, quality = tag_name.split("_")
        return int(dim), int(quality)

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
        self.metadata().unlink(missing_ok=True)
        self.metadata().touch()

    def merged_videos_dir(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.s_videos.value)

    def merged_videos_from(self, subject_id: int) -> Path:
        return self.merged_frames().joinpath(f"result_from_{subject_id}.mp4")

    def merged_frames(self) -> Path:
        return self.__root.joinpath(WorkspaceStr.s_frames.value)

    def merged_frames_from(self, subject_id: int):
        return self.merged_frames().joinpath(WorkspaceStr.dst_video.value + str(subject_id))

    def mask_frames_from(self, subject_id: int):
        return self.merged_frames_from(subject_id).joinpath(WorkspaceStr.mask.value)

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
        import shutil

        aligned_dir = self.__root.joinpath(WorkspaceStr.aligned.value)
        if aligned_dir.exists():
            shutil.rmtree(aligned_dir)
        aligned_dir.mkdir(exist_ok=True)
        for tag in self.__root.glob(".tag*"):
            tag.unlink()

    def clean(self):
        import shutil

        for directory in [self.__root.joinpath(WorkspaceStr.s_videos.value),
                          self.__root.joinpath(WorkspaceStr.s_frames.value),
                          self.__root.joinpath(WorkspaceStr.frames.value)]:
            if directory.exists():
                shutil.rmtree(directory)
        for tag in self.__root.glob(".tag*"):
            tag.unlink()
        self.metadata().unlink(missing_ok=True)
