import argparse
from pathlib import Path
from typing import Dict, Union


def args_parser() -> Dict[str, any]:
    parser = argparse.ArgumentParser(
        prog="argparse, args_parser.py",
        description="Args parser for automatic mass face swapping",
    )

    subparsers = parser.add_subparsers(title="Action", dest="action")

    to_subject_action_parser(subparsers)

    update_subject_workspace_action_parser(subparsers)

    clean_action_parser(subparsers)

    extract_action_parser(subparsers)

    pack_action_parser(subparsers)

    unpack_action_parser(subparsers)

    pretrain_action_parser(subparsers)

    swap_action_parser(subparsers)

    face_swap_benchmark_action_parser(subparsers)

    similarity_score_benchmark_action_parser(subparsers)

    return vars(parser.parse_args())


def similarity_score_benchmark_action_parser(subparsers):
    parser_benchmark = subparsers.add_parser("similarity_score_benchmark", help="Perform Data augmentation for " +
                                                                                "similarity score" +
                                                                                " benchmarking.")
    parser_benchmark.add_argument("--similarity_score_benchmark_dir", type=Path)
    parser_benchmark.add_argument("--number_data_augmentation_loop", type=int)


def face_swap_benchmark_action_parser(subparsers):
    parser_benchmark = subparsers.add_parser("face_swap_benchmark",
                                             help="Perform benchmarking on all available SAEHD models.")
    parser_benchmark.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_benchmark.add_argument("--subject_src_id", type=int, help="Id of source subject to perform benchmark.")
    parser_benchmark.add_argument("--subject_dst_id", type=int, help="Id of destination subject to perform benchmark.")
    parser_benchmark.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_benchmark.add_argument("--benchmark_output_path_results", type=Path, help="Path to save the benchmark " +
                                                                                     "results, the folder need to be " +
                                                                                     "empty and does not need to " +
                                                                                     "exists only the parent folder " +
                                                                                     "needs to.")
    parser_benchmark.add_argument("--iteration_goal", type=int, help="Integer of the last iteration to be performed.")
    parser_benchmark.add_argument("--delta_iteration", type=int, help="Integer of the iterations between each " +
                                                                      "benchmarking.")


def swap_action_parser(subparsers):
    parser_swap = subparsers.add_parser("swap", help="Face swapping")
    parser_swap.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_swap.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_swap.add_argument("--model_name", type=str, default="", help="Name of the SAEHD model")
    parser_swap.add_argument("--iteration_goal", type=int, default=None, help="Iteration to reach " +
                                                                                          " for each face swapping " +
                                                                                          "operation")


def pretrain_action_parser(subparsers):
    parser_pretrain = subparsers.add_parser("pretrain", help="Pretrain model")
    parser_pretrain.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_pretrain.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_pretrain.add_argument("--model_name", type=str, default="", help="Name of the SAEHD model")
    parser_pretrain.add_argument("--model_dir_backup", type=Path, help="Path to the model backup directory")


def unpack_action_parser(subparsers):
    parser_unpack = subparsers.add_parser("unpack", help="Unpack packed faces")
    parser_unpack.add_argument("subjects_dir", type=Path, help="Path to the subjects directory")


def pack_action_parser(subparsers):
    parser_pack = subparsers.add_parser("pack", help="Pack extracted faces")
    parser_pack.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")


def extract_action_parser(subparsers):
    parser_extract = subparsers.add_parser("extract", help="Extract faces from subjects")
    parser_extract.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_extract.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_extract.add_argument("--png_quality", type=int, help="Quality of the extracted images")


def update_subject_workspace_action_parser(subparsers):
    parser_update_wrk = subparsers.add_parser("update_wrk", help="Update workspace")
    parser_update_wrk.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")


def clean_action_parser(subparsers):
    parser_clean = subparsers.add_parser("clean", help="Clean workspace")
    parser_clean.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_clean.add_argument("--redo_merged_workspace", action='store_true')


def to_subject_action_parser(subparsers):
    parser_to_subject = subparsers.add_parser("to_subject", help="Convert videos to subjects")
    parser_to_subject.add_argument("--videos_dir", type=Path, help="Path to the videos directory")
    parser_to_subject.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
