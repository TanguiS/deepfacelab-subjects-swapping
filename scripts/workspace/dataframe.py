from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

from scripts.workspace import workspace


def dataframe_value(frame: Path, video: Path, label: bool, original: Optional[Path] = None) -> Dict[str, any]:
    return {
        'frame': frame,
        'video': video,
        'label': label,
        'original': original
    }


def put_item(
        container:  List[Dict[str, any]],
        datas: List[Path],
        root: Path,
        video: Path,
        label: bool,
        original: Optional[Path] = None
) -> None:
    for frame in datas:
        item = dataframe_value(
            frame.relative_to(root),
            video.relative_to(root),
            label,
            original.relative_to(root) if original else None
        )
        container.append(item)


def create(subjects_dir: Path, output_pickle_dataframe: Path) -> None:
    subjects = workspace.load_subjects(subjects_dir)
    total = len(subjects) * (len(subjects) - 1)
    progress_bar = tqdm(
        total=total,
        desc="Indexing aligned frames",
        unit=" subject"
    )

    indexer = []

    for subject_src in subjects:
        aligned_frames = [frame for frame in subject_src.frame.original.aligned_dir().glob("*.jpg")]

        put_item(indexer, aligned_frames, subjects_dir, subject_src.video.original_video(), False)

        for subject_dst in subjects:
            if subject_src == subject_dst:
                continue

            aligned_merged_frames = [
                frame for frame in subject_src.frame.merged.frames_dir_from(subject_dst.id()).glob("*.png")
            ]

            put_item(
                indexer,
                aligned_merged_frames,
                subjects_dir,
                subject_src.video.merged_videos_from_subject_id(subject_dst.id()),
                True,
                subject_src.video.original_video()
            )
            progress_bar.update(1)

    progress_bar.close()

    print("Creating DataFrame...")
    df = pd.DataFrame(indexer).set_index('frame')

    print(f"Saving dataframe to {output_pickle_dataframe}...")
    df.to_pickle(str(output_pickle_dataframe))

    print(f"Dataframe information :\n - number real frames : {sum(df['label'] == False)}\n - number fake frames : {sum(df['label'] == True)}")


if __name__ == "__main__":
    sub_path = Path('D:\\storage-photos\\subjects')
    out_path = Path('D:\\storage-photos\\subjects\\output.pkl')

    create(sub_path, out_path)
