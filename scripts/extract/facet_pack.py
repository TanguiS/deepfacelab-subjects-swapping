from pathlib import Path
from typing import List

from scripts.Subject import Subject


def pack(src_dir: Path) -> None:
    from samplelib import PackedFaceset
    PackedFaceset.pack(src_dir)


def launch(subjects: List[Subject], to_pack_dir: Path) -> None:
    from scripts.util import cp_aligned_frames, hard_clean_dir, smooth_clean_dir

    if not to_pack_dir.exists():
        raise NotADirectoryError
    hard_clean_dir(to_pack_dir)
    cp_aligned_frames(subjects, to_pack_dir)
    pack(to_pack_dir)
    smooth_clean_dir(to_pack_dir, True)
