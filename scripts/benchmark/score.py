import csv
from pathlib import Path
from typing import Generator, List


def frames_to_evaluate(benchmark_dir: Path) -> Generator[Path, Path]:
    frames = {frames for frames in benchmark_dir.rglob("*.png")}
    reference_face = benchmark_dir.joinpath("reference_face.png")
    frames.remove(reference_face)
    for frame in frames:
        yield reference_face, frame


def write_threshold_results(benchmark_dir: Path, threshold: List[float]) -> None:
    benchmark_csv = benchmark_dir.joinpath("benchmark.csv")
    with open(benchmark_csv, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    lines[0].append("Threshold")
    for i, line in enumerate(lines[1:]):
        line.append('{:.3f}'.format(threshold[i]))

    with open(benchmark_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lines)


