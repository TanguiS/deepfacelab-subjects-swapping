import csv
import random
from pathlib import Path
from typing import Generator, List, Tuple


def frames_to_evaluate(benchmark_dir: Path) -> Generator[Tuple[Path], None, None]:
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


if __name__ == '__main__':
    benchmark_dir = Path("/media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/benchmark")
    thres = []
    for ref, frame in frames_to_evaluate(benchmark_dir):
        print(f"ref : {ref}, frame : {frame}")
        rd = random.random()
        print(f"  -> random value affected : {rd}")
        thres.append(rd)
    # write_threshold_results(benchmark_dir, thres)

