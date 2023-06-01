import csv
import random
from pathlib import Path
from typing import Generator, List, Tuple


def frames_to_evaluate(benchmark_dir: Path) -> Generator[Tuple[Path], None, None]:
    frames = {frames for frames in benchmark_dir.rglob("*.jpg")}
    reference_face = benchmark_dir.joinpath("reference_face.png")
    try:
        frames.remove(reference_face)
    except:
        pass
    for frame in frames:
        yield reference_face, frame


def get_specs(frames_path: Path) -> Tuple[str, int]:
    specs = frames_path.as_posix()
    specs = specs.split("/")
    return specs[-3], int(specs[-2])


def write_threshold_results(benchmark_dir: Path, thresholds: List[Tuple[str, int, float]]) -> None:
    benchmark_csv = benchmark_dir.joinpath("benchmark.csv")
    with open(benchmark_csv, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    lines[0].append("Threshold")
    for model, it, threshold in thresholds:
        for line in lines[1:]:
            if line[0] == model and int(line[1]) == it:
                line.append(threshold)
                break

    with open(benchmark_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lines)


def write_similarity_scores_bench(benchmark_dir: Path, scores: Tuple[str, float]) -> None:
    benchmark_csv = benchmark_dir.joinpath("benchmark.csv")
    if not benchmark_csv.exists():
        benchmark_csv.touch()
    with open(benchmark_csv, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([scores[0], scores[1]])


if __name__ == '__main__':
    benchmark_dir = Path("/media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/similqrity_scores_benchmark")
    thres = []
    for ref, frame in frames_to_evaluate(benchmark_dir):
        rd = random.random()
        write_similarity_scores_bench(benchmark_dir, [frame.name, rd])
