from pathlib import Path
from typing import List

from scripts.SubjectLoader import Subject


def cp_aligned_frames(subjects: List[Subject], dst_dir: Path) -> None:
    from scripts.env.workspace import WorkspaceStr
    import shutil

    hard_clean_dir(dst_dir)
    for subject in subjects:
        for src_file in subject.aligned().glob('*'):
            tmp = dst_dir / (WorkspaceStr.subject.value + str(subject.id()) + "_" + src_file.stem + src_file.suffix)
            shutil.copy(src_file, tmp)


def hard_clean_dir(dst_dir: Path) -> None:
    import shutil

    shutil.rmtree(dst_dir)
    dst_dir.mkdir(exist_ok=True)


def smooth_clean_dir(dst_dir: Path, pak_ok: bool = False) -> None:
    for img in dst_dir.glob('*'):
        if (pak_ok and img.suffix == ".pak") or not img.exists():
            continue
        img.unlink()
