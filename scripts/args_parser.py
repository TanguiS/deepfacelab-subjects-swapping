import argparse
from pathlib import Path
from typing import Dict, Union, List


def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, args_parser.py",
        description="Args parser for automatic face swapping scripts according to a end condition",
    )

    parser.add_argument(
        "-a", "--action",
        choices=["to_subject", "extract", "pack", "unpack", "pretrain", "swap", "clean"],
        default="extract",
        help="Action made by the script :\n   -> extract : extract the whole face of each subject\n   -> pack : pack "
             "the extracted faces in order to pretrain\n<default::extract>"
    )

    parser.add_argument(
        "-c", "--clean",
        action="store_true",
        help="clean subjects results"
    )

    parser.add_argument(
        "--png_quality",
        type=int,
        default=None
    )

    parser.add_argument(
        "--dim_output_faces",
        type=int,
        default=None
    )

    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Dir to the model to use"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Name of the SAEHD model used, will be asked with a list of model or create one if not given"
    )

    parser.add_argument(
        "--videos_dir",
        type=Path
    )

    parser.add_argument(
        "--model_dir_backup",
        type=Path
    )

    parser.add_argument(
        "--subjects_dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--subject_id_src",
        type=int
    )

    parser.add_argument(
        "--subject_id_dst",
        type=int
    )

    parser.add_argument(
        '--gpu_indexes',
        type=str
    )

    parser.add_argument(
        '--silent_start',
        type=bool,
        default=False
    )

    return vars(parser.parse_args())
