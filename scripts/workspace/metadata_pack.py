import json
from typing import List, Union, Dict, Optional

from tqdm import tqdm

from scripts.Subject import Subject


def value(label: str, split: str, original: Union[str, any] = None) -> Dict[str, Optional[str]]:
    return {
        'label': label,
        'split': split,
        'original': original
    }


def launch(subjects: List[Subject]) -> None:
    for subject_src in tqdm(subjects, total=len(subjects), desc="creating metadata", miniters=1.0, unit="subject"):
        subject_src.reset_metadata()
        data = {}

        original = subject_src.original_video().name
        data[original] = value("REAL", "train")

        for subject_dst in subjects:
            if subject_dst == subject_src:
                continue
            fake = subject_src.merged_videos_from(subject_dst.id())
            tmp = fake.parent
            fake = str(tmp.name) + "/" + str(fake.name)
            original = subject_dst.original_video().name
            data[fake] = value("FAKE", "train", original)

        with open(subject_src.metadata(), 'w') as file:
            json.dump(data, file)
