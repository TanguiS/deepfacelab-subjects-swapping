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
        choices=["to_subject", "update_wrk", "extract", "pack", "unpack", "pretrain", "swap", "clean"],
        default="extract",
        help="Action made by the script :\n   -> extract : extract the whole face of each subject\n   -> pack : pack "
             "the extracted faces in order to pretrain\n<default::extract>"
    )

    parser.add_argument(
        "--png_quality",
        type=int,
        default=None,
        help="Quality of the extracted images from the raw videos."
    )

    parser.add_argument(
        "--dim_output_faces",
        type=int,
        default=None,
        help="Dim of the extracted output faces."
    )

    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Dir to the models. Can have more than one model."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Name of the SAEHD model used, will be asked with a list of model from the --model_dir arg or create " +
             "one if empty"
    )

    parser.add_argument(
        "--videos_dir",
        type=Path,
        help="Path to the videos directory to create subjects from them."
    )

    parser.add_argument(
        "--model_dir_backup",
        type=Path,
        help="Path to the backup for the pretrained model, used for face swapping. If None -> no backup at the end."
    )

    parser.add_argument(
        "--subjects_dir",
        type=Path,
        required=True,
        help="Path to the subjects directory."
    )

    parser.add_argument(
        '--gpu_indexes',
        type=str,
        help="Not your concern for now."
    )

    return vars(parser.parse_args())
