from pathlib import Path
from typing import Tuple, Union, List

from scripts.workspace.WorkspaceEnum import WorkspaceStr


def parse_subject_tags(tag: Path) -> Tuple[int, int]:
    tag_name = tag.name[5:]
    dim, quality = tag_name.split("_")
    return int(dim), int(quality)


class Subject:
    def __init__(self, subject_path: Path, dim: Union[int, any] = None, quality: Union[int, any] = None) -> None:
        self.__root = subject_path

        if dim is None or quality is None:
            available_tags = self.__find_available_tags()
            if len(available_tags) == 0:
                raise ValueError("No available tags found in the subject folder.")

            elif len(available_tags) == 1:
                print(f"Found 1 tag : {available_tags[0]}")
                self.__dim, self.__quality = parse_subject_tags(available_tags[0])
            else:
                print("/!\\ More than 1 tag is present, clean the workspace.")
                raise Exception("Feature not implemented.")
        else:
            self.__dim = dim
            self.__quality = quality

        self.__original_tag = subject_path.joinpath(f".tag_{self.__dim}_{self.__quality}")

        self.__video = Video(self)
        self.__frame = Frame(self, self.__original_tag)
        self.__clean = Clean(self)

    def __find_available_tags(self) -> List[Path]:
        tags = [file for file in self.__root.glob(".tag_*")]
        return tags

    def id(self) -> int:
        return int(self.__root.as_posix().split("_")[-1])

    def specs(self) -> Tuple[int, int]:
        return self.__dim, self.__quality

    def root_dir(self) -> Path:
        return self.__root

    def __str__(self) -> str:
        return f"subject id : {self.id()}"

    @property
    def video(self):
        return self.__video

    @property
    def frame(self):
        return self.__frame

    @property
    def clean(self):
        return self.__clean


class Video:
    def __init__(self, subject: Subject) -> None:
        super().__init__()
        self.__subject = subject

    def original_video(self) -> Path:
        output = [video for video in self.__subject.root_dir().glob(WorkspaceStr.videos_pattern.value)]
        if len(output) > 1:
            raise KeyError(f"Error, multiple videos found for pattern : {WorkspaceStr.videos_pattern.value}, videos found : " +
                           f"{output} in subject id {self.__subject.id()}")
        if len(output) == 0:
            raise KeyError(f"Error, no original video found in subject id : {self.__subject.id()}.")
        return output[0]

    def merged_videos_dir(self) -> Path:
        return self.__subject.root_dir().joinpath(WorkspaceStr.merged_videos.value)

    def merged_videos_from_subject_id(self, subject_id: int) -> Path:
        return self.merged_videos_dir().joinpath(f"result_from_{subject_id}.mp4")


class Frame:
    def __init__(self, subject: Subject, tag: Path) -> None:
        super().__init__()
        self.__subject = subject

        self.__original = Original(self.__subject, tag)
        self.__merged = Merged(self.__subject)
        self.__face = Face(self.__subject)

    @property
    def original(self):
        return self.__original

    @property
    def merged(self):
        return self.__merged

    @property
    def face(self):
        return self.__face


class Original:
    def __init__(self, subject: Subject, tag: Path) -> None:
        super().__init__()
        self.__subject = subject

        self.__original_tag = tag

    def frames_dir(self) -> Path:
        return self.__subject.root_dir().joinpath(WorkspaceStr.frames.value)

    def aligned_dir(self) -> Path:
        return self.__subject.root_dir().joinpath(WorkspaceStr.frames.value).joinpath(WorkspaceStr.aligned.value)

    def align_extract_done(self) -> None:
        if not self.__original_tag.exists():
            self.__original_tag.touch()

    def is_align_extract_done(self) -> bool:
        return self.__original_tag.exists()


class Merged:
    def __init__(self, subject: Subject) -> None:
        super().__init__()
        self.__subject = subject

        self.__tag_merged = ".done"

    def frames_dir(self) -> Path:
        return self.__subject.root_dir().joinpath(WorkspaceStr.merged_frames.value)

    def frames_dir_from(self, subject_id: int) -> Path:
        return self.frames_dir().joinpath(WorkspaceStr.merged_dst.value + str(subject_id))

    def mask_frames_dir_from(self, subject_id: int):
        return self.frames_dir_from(subject_id).joinpath(WorkspaceStr.mask.value)

    def merging_done_from(self, subject_id: int) -> None:
        self.frames_dir_from(subject_id).joinpath(self.__tag_merged).touch(exist_ok=True)

    def is_merging_from_is_done(self, subject_id: int) -> bool:
        return self.frames_dir_from(subject_id).joinpath(self.__tag_merged).exists()


class Face:
    def __init__(self, subject: Subject) -> None:
        super().__init__()
        self.__subject = subject

    def frames_dir(self) -> Path:
        return self.__subject.root_dir().joinpath(WorkspaceStr.frames.value).joinpath(WorkspaceStr.face.value)

    def frames_dir_from(self, subject_id: int) -> Path:
        return self.__subject.frame.merged.frames_dir_from(subject_id).joinpath(WorkspaceStr.face.value)

    def extracting_done_from(self, subject_id: int, max_shape: int) -> None:
        tag = self.frames_dir_from(subject_id).joinpath(
            f".tag_{max_shape}.done")
        if not tag.exists():
            tag.touch()

    def is_extracting_done_from(self, subject_id: int, max_shape: int) -> bool:
        tag = self.frames_dir_from(subject_id).joinpath(
            f".tag_{max_shape}.done")
        return tag.exists()

    def extracting_done(self, max_shape: int) -> None:
        tag = self.frames_dir().joinpath(f".tag_{max_shape}.done")
        if not tag.exists():
            tag.touch()

    def is_extracting_done(self, max_shape: int) -> bool:
        tag = self.frames_dir().joinpath(f".tag_{max_shape}.done")
        return tag.exists()


class Clean:
    def __init__(self, subject: Subject) -> None:
        super().__init__()
        self.__subject = subject

    def clean_alignment(self) -> None:
        import shutil

        aligned_dir = self.__subject.frame.original.aligned_dir()
        if aligned_dir.exists():
            shutil.rmtree(aligned_dir)
        aligned_dir.mkdir(exist_ok=True)
        for tag in self.__subject.root_dir().glob(".tag*"):
            tag.unlink()

    def clean_mask_from(self, subject_id: int) -> None:
        import shutil

        masks_dir = self.__subject.frame.merged.mask_frames_dir_from(subject_id)
        if masks_dir.exists():
            shutil.rmtree(masks_dir)

    def clean_merged(self) -> None:
        import shutil

        if self.__subject.frame.merged.frames_dir().exists():
            shutil.rmtree(self.__subject.frame.merged.frames_dir())
        if self.__subject.video.merged_videos_dir().exists():
            shutil.rmtree(self.__subject.video.merged_videos_dir())

    def clean_face(self) -> None:
        import shutil

        faces = [
            face_folder.joinpath(WorkspaceStr.face.value)
            for face_folder in self.__subject.frame.merged.frames_dir().iterdir()
        ]
        faces.append(self.__subject.frame.face.frames_dir())

        for face in faces:
            shutil.rmtree(face)

    def clean_all(self):
        import shutil

        for directory in [self.__subject.root_dir().joinpath(WorkspaceStr.merged_videos.value),
                          self.__subject.root_dir().joinpath(WorkspaceStr.merged_frames.value),
                          self.__subject.root_dir().joinpath(WorkspaceStr.frames.value)]:
            if directory.exists():
                shutil.rmtree(directory)
        for tag in self.__subject.root_dir().glob(".tag*"):
            tag.unlink()
