from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Union

import cv2
import pandas as pd
import hashlib

from pandas import DataFrame
from tqdm import tqdm

from scripts.Subject import Subject
from scripts.extract.random_data_augmentation import specs_face_data_augmentation, decode_specs_data_augmentation
from scripts.workspace import workspace
from scripts.workspace.WorkspaceEnum import WorkspaceStr


def save_asset(indexer: List[Dict[str, any]], output_pickle_dataframe_path: Path) -> None:
    print("Creating DataFrame...")
    df = pd.DataFrame(indexer).set_index('frame')
    print(f"Saving dataframe to {output_pickle_dataframe_path}...")
    df.to_pickle(str(output_pickle_dataframe_path))
    print(
        f"Dataframe information :\n" +
        f" - number real frames : {sum(df['label'] == False)}\n" +
        f" - number fake frames : {sum(df['label'] == True)}"
    )


def load_asset(output_pickle_dataframe_path: Path, subjects_dir: Path
               ) -> Tuple[Set[str], List[Dict[str, any]], List[Subject]]:
    subjects = workspace.load_subjects(subjects_dir)
    indexer = []
    check_indexer = set()
    if output_pickle_dataframe_path.exists() and output_pickle_dataframe_path.is_file():
        df: DataFrame = pd.read_pickle(output_pickle_dataframe_path)
        indexer = df.reset_index().to_dict(orient='records')
        check_indexer = set(df['sha256'])
        print(
            f"Dataframe information :\n" +
            f" - number real frames : {sum(df['label'] == False)}\n" +
            f" - number fake frames : {sum(df['label'] == True)}"
        )
    return check_indexer, indexer, subjects


def sha256(image_path: Path) -> str:
    sha256_hash = hashlib.sha256()

    with open(image_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def dataframe_value(
        frame: Union[Path, any],
        video: Union[Path, any],
        label: Union[bool, any],
        img_face_size: int,
        original: Optional[Union[Path, any]] = None,
        sha256_value: Optional[Union[Path, any]] = None
) -> Dict[str, any]:
    return {
        'frame': frame,
        'video': video,
        'label': label,
        'original': original,
        'sha256': sha256_value,
        'left': 0,
        'top': 0,
        'right': img_face_size,
        'bottom': img_face_size
    }


def put_items_with_similarity_check(
        container: List[Dict[str, any]],
        check_container: Set[str],
        datas: List[Path],
        root: Path,
        video: Path,
        label: bool,
        img_face_size: int,
        original: Optional[Path] = None
) -> None:
    for frame in datas:

        if any(d['frame'] == frame.relative_to(root) for d in container):
            continue

        sha256_result = sha256(frame)
        if sha256_result in check_container:
            continue

        item = dataframe_value(
            frame.relative_to(root),
            video.relative_to(root),
            label,
            img_face_size,
            original.relative_to(root) if original else None,
            sha256_result
        )
        container.append(item)
        check_container.add(sha256_result)


def add_data_augmentation_dataframe(
        subjects_dir: Path,
        container: List[Dict[str, any]],
        check_container: Set[str]
) -> None:
    print("Loading face to extract...")

    face_frames = []
    for label_dir in (WorkspaceStr.fake_aug.value, WorkspaceStr.real_aug.value):
        face_frames.extend([
            frame for frame in subjects_dir.joinpath(
                WorkspaceStr.augmentation.value).joinpath(
                label_dir).joinpath(
                WorkspaceStr.face.value).glob("*.png")
        ])

    total = len(face_frames)
    for face in tqdm(face_frames, total=total, desc="Indexing face frames from data augmentation", unit=" image "):

        if any(d['frame'] == face.relative_to(subjects_dir) for d in container):
            continue

        original, label = specs_face_data_augmentation(face)
        try:
            original, label = decode_specs_data_augmentation(subjects_dir, original, label)
        except ValueError as e:
            print(e)
            continue

        sha256_result = sha256(face)
        if sha256_result in check_container:
            continue

        item = dataframe_value(
            face.relative_to(subjects_dir),
            face.relative_to(subjects_dir),
            label,
            sha256_value=sha256_result
        )
        container.append(item)
        check_container.add(sha256_result)


def create(subjects_dir: Path, output_pickle_dataframe_path: Path) -> None:
    check_indexer, indexer, subjects = load_asset(output_pickle_dataframe_path, subjects_dir)

    total = len(subjects) * (len(subjects) - 1) + len(subjects)
    progress_bar = tqdm(
        total=total,
        desc="Indexing face frames from subject",
        unit=" subject"
    )

    putter = put_items_with_similarity_check

    for subject_src in subjects:
        face_frames = [frame for frame in subject_src.frame.face.frames_dir().glob("*.png")]
        img_face_size = cv2.imread(face_frames[0], cv2.IMREAD_UNCHANGED).shape[1]

        putter(
            indexer, check_indexer, face_frames, subjects_dir, subject_src.video.original_video(), False, img_face_size
        )
        del img_face_size
        progress_bar.update(1)

        for subject_dst in subjects:
            if subject_src == subject_dst:
                continue

            merged_face_frames = [
                frame for frame in subject_src.frame.face.frames_dir_from(subject_dst.id()).glob("*.png")
            ]
            img_face_size = cv2.imread(merged_face_frames[0], cv2.IMREAD_UNCHANGED).shape[1]

            putter(
                indexer,
                check_indexer,
                merged_face_frames,
                subjects_dir,
                subject_src.video.merged_videos_from_subject_id(subject_dst.id()),
                True,
                img_face_size,
                subject_src.video.original_video()
            )
            del img_face_size
            progress_bar.update(1)

        save_asset(indexer, output_pickle_dataframe_path)

    progress_bar.close()

    add_data_augmentation_dataframe(subjects_dir, indexer, check_indexer)

    save_asset(indexer, output_pickle_dataframe_path)


if __name__ == "__main__":
    sub_path = Path('D:\\storage-photos\\subjects')
    out_path = Path('D:\\storage-photos\\subjects\\output.pkl')

    create(sub_path, out_path)
