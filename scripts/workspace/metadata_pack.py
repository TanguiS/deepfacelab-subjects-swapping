import json
from pathlib import Path
from typing import List

from scripts.Subject import Subject


def metadata_pack(subjects: List[Subject], output_dir_metapack: Path) -> None:
    if output_dir_metapack.is_file():
        raise IOError("Error: output dir for metapack is a file.")
    output_dir_metapack.mkdir(exist_ok=True)

    metapack = output_dir_metapack.joinpath("metadata.json")
    if metapack.exists():
        metapack.unlink()
    metapack.touch()

    data = {}
    for i, subject_src in enumerate(subjects):
        original = subject_src.original_video()
        value = {
            'label': "REAL",
            'split': "train",
            'original': None
        }
        data[original] = value
        for j, subject_dst in enumerate(subjects):
            if i == j:
                continue
            original = subject_dst.original_video()
            fake = subject_src.merged_videos_from(subject_dst.id())
            value = {
                'label': "FAKE",
                'split': "train",
                'original': original
            }
            data[fake] = value

        with open(metapack, 'w') as file:
            json.dump(data, file)


