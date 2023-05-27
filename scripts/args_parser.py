import argparse
from pathlib import Path
from typing import Dict, Union, List


def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, args_parser.py",
        description="Args parser for automatic face swapping scripts according to an end condition",
    )

    subparsers = parser.add_subparsers(title="Action", dest="action")

    # Create the sub-parser for the 'to_subject' action
    to_subject_action_parser(subparsers)

    # Create the sub-parser for the 'update_wrk' action
    parser_update_wrk = subparsers.add_parser("update_wrk", help="Update workspace")
    parser_update_wrk.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_update_wrk.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_update_wrk.add_argument("--png_quality", type=int, help="Quality of the extracted images")

    # Create the sub-parser for the 'clean' action
    clean_action_parsser(subparsers)

    # Create the sub-parser for the 'extract' action
    parser_extract = subparsers.add_parser("extract", help="Extract faces from subjects")
    parser_extract.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_extract.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_extract.add_argument("--png_quality", type=int, help="Quality of the extracted images")

    # Create the sub-parser for the 'pack' action
    parser_pack = subparsers.add_parser("pack", help="Pack extracted faces")
    parser_pack.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_pack.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_pack.add_argument("--png_quality", type=int, help="Quality of the extracted images")

    # Create the sub-parser for the 'unpack' action
    parser_unpack = subparsers.add_parser("unpack", help="Unpack packed faces")
    parser_unpack.add_argument("subjects_dir", type=Path, help="Path to the subjects directory")

    # Create the sub-parser for the 'pretrain' action
    parser_pretrain = subparsers.add_parser("pretrain", help="Pretrain model")
    parser_pretrain.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_pretrain.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_pretrain.add_argument("--png_quality", type=int, help="Quality of the extracted images")
    parser_pretrain.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_pretrain.add_argument("--model_name", type=str, default="", help="Name of the SAEHD model")
    parser_pretrain.add_argument("--model_dir_backup", type=Path, help="Path to the model backup directory")

    # Create the sub-parser for the 'swap' action
    parser_swap = subparsers.add_parser("swap", help="Face swapping")
    parser_swap.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_swap.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_swap.add_argument("--png_quality", type=int, help="Quality of the extracted images")
    parser_swap.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_swap.add_argument("--model_name", type=str, default="", help="Name of the SAEHD model")

    parser.add_argument("--gpu_indexes", type=str, help="Not your concern for now.")

    return vars(parser.parse_args())


def clean_action_parsser(subparsers):
    parser_clean = subparsers.add_parser("clean", help="Clean workspace")
    parser_clean.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_clean.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_clean.add_argument("--png_quality", type=int, help="Quality of the extracted images")


def to_subject_action_parser(subparsers):
    parser_to_subject = subparsers.add_parser("to_subject", help="Convert videos to subjects")
    parser_to_subject.add_argument("--videos_dir", type=Path, help="Path to the videos directory")
    parser_to_subject.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
