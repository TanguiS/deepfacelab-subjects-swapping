from pathlib import Path
from typing import Optional

from core import osex
from mainscripts import VideoEd


def video_to_frames(input_video: Path, output_dir: Path, name_stem: Optional[str] = None) -> None:
    osex.set_process_lowest_prio()
    VideoEd.extract_video(
        input_file=input_video,
        output_dir=output_dir,
        output_ext='png',
        fps=0,
        output_name_stem=name_stem
    )
