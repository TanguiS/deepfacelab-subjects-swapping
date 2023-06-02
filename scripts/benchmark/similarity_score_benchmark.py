"""
import random
from pathlib import Path
from typing import Tuple, Generator

from scripts.Subject import Subject
from scripts.benchmark.face_quality_benchmark import get_score_from_fqa_service
from scripts.benchmark.util import clean_output_csv, path_to_b64, write_bench_generic
from scripts.workspace.workspace import load_subjects


def select_frames(subject: Subject, number_image_to_evaluate_per_subject: int = 1) -> Generator[Tuple[Path, Path], None, None]:
    reference_frame = select_reference_frame(subject)

    frames_to_evaluate = []
    merged_frames = [folder for folder in subject.merged_frames().iterdir()]
    for merged_frames_from_subject in merged_frames:
        merged_frames = [img for img in merged_frames_from_subject.glob("*.png")]
        keep_indexes = []
        for _ in range(0, number_image_to_evaluate_per_subject):
            keep_index = random.randint(0, len(merged_frames))
            while keep_index in keep_indexes:
                keep_index = random.randint(0, len(merged_frames))
            keep_indexes.append(keep_index)
            yield reference_frame, frames_to_evaluate


def select_reference_frame(subject: Subject) -> Path:
    reference_frame = [img for img in subject.original_frames().glob("*.png")]
    keep_index = random.randint(0, len(reference_frame))
    reference_frame = reference_frame[keep_index]
    return reference_frame


def get_subjects_id_dst(merged_frames_from_subject: Path) -> int:
    parent_folder_name = merged_frames_from_subject.parent.name
    return int(parent_folder_name.split("_")[-1])


def launch_benchmark_evaluation(subjects_dir: Path, output_csv_name: str) -> None:
    benchmark_csv = clean_output_csv(subjects_dir, output_csv_name)
    subjects = load_subjects(subjects_dir)
    write_bench_generic(benchmark_csv, ["Subject_id_src", "Subject_id_dst", "Quality_score", "Similarity_score"])
    for subject in subjects:
        for reference_file, frames_to_evaluate in select_frames(subject, 5):
            similarity_score = random.uniform(0, 1)
            quality_score = get_score_from_fqa_service(path_to_b64(frames_to_evaluate))
            write_bench_generic(benchmark_csv, [
                str(subject.id()),
                str(get_subjects_id_dst(frames_to_evaluate)),
                str(quality_score),
                str(similarity_score)
            ])


if __name__ == '__main__':
    subjects_dir = Path("")
    launch_benchmark_evaluation(subjects_dir, "benchmark_deepfake_evaluation.csv")
"""
