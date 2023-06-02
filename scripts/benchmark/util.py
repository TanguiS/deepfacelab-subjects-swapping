import base64
import csv
from pathlib import Path
from typing import Generator, Tuple, List


def frames_to_evaluate_from_benchmark_dir(benchmark_dir: Path) -> Generator[Tuple[Path, Path], None, None]:
    frames = {frames for frames in benchmark_dir.rglob("*.jpg")}
    reference_face = benchmark_dir.joinpath("reference_face.png")
    try:
        frames.remove(reference_face)
    except KeyError:
        print("No reference file found !")
        pass
    for frame in frames:
        yield reference_face, frame


def get_model_iteration_reached(frame_path: Path) -> Tuple[str, int]:
    specs = frame_path.as_posix()
    specs = specs.split("/")
    return specs[-3], int(specs[-2])


def write_bench_generic(csv_file: Path, data: List[str]) -> None:
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def clean_output_csv(output_dir: Path, output_csv_name: str) -> Path:
    file_csv = output_dir.joinpath(output_csv_name)
    if file_csv.exists():
        file_csv.unlink()
    file_csv.touch()
    return file_csv


def path_to_b64(frame_path: Path) -> str:
    with open(frame_path, "rb") as frame_file:
        frame_b64 = base64.b64encode(frame_file.read())
    frame_b64 = frame_b64.decode('ascii')
    return frame_b64
