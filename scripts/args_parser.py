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

    raw_extract_action_parser(subparsers)

    swap_extract_action_parser(subparsers)

    pack_action_parser(subparsers)

    unpack_action_parser(subparsers)

    pretrain_action_parser(subparsers)

    swap_auto_action_parser(subparsers)

    swap_flexible_train_action_parser(subparsers)

    swap_flexible_merge_action_parser(subparsers)

    clean_flexible_train_action_parser(subparsers)

    face_swap_benchmark_action_parser(subparsers)

    dataframe_action_parser(subparsers)

    extract_face_from_subject_action_parser(subparsers)

    extract_face_from_video_data_augmentation_action_parser(subparsers)

    return vars(parser.parse_args())


def extract_face_from_video_data_augmentation_action_parser(subparsers):
    extract_face = subparsers.add_parser("extract_face_from_video_data_augmentation",
                                         help="Extract frames from all videos contained in the data augmentation" +
                                              " folder, Warning, only video and no sub_folder.")
    extract_face.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    extract_face.add_argument("--model_dir", type=Path, help="Path to the onxx models directory for YuNet class")
    extract_face.add_argument("--input_shape", type=int,
                              help="Square shape of the input YuNet class for face extraction")
    extract_face.add_argument("--max_shape", type=int,
                              help="Shape of the output faces, should be less than shape")


def extract_face_from_subject_action_parser(subparsers):
    parser_extract_face = subparsers.add_parser("extract_face_from_subject", help="Extract faces from merged" +
                                                                                  " and original frames folder" +
                                                                                  " for every subject")
    parser_extract_face.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_extract_face.add_argument("--model_dir", type=Path, help="Path to the onxx models directory for YuNet class")
    parser_extract_face.add_argument("--input_shape", type=int,
                                     help="Square shape of the input YuNet class for face extraction")
    parser_extract_face.add_argument("--max_shape", type=int,
                                     help="Shape of the output faces, should be less than shape")


def clean_flexible_train_action_parser(subparsers):
    parser_clean_flex_train = subparsers.add_parser("clean_flexible_train", help="Clean all trained models.")
    parser_clean_flex_train.add_argument("--model_dir", type=Path, help="Path to the models directory")


def swap_flexible_merge_action_parser(subparsers):
    parser_swap_flex_merge = subparsers.add_parser("swap_flexible_merge", help="Face swapping with flexible merger")
    parser_swap_flex_merge.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_swap_flex_merge.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_swap_flex_merge.add_argument("--model_name", type=str, default="", help="Name of the SAEHD model")


def swap_flexible_train_action_parser(subparsers):
    parser_swap_flex_train = subparsers.add_parser("swap_flexible_train", help="Face swapping with flexible merger")
    parser_swap_flex_train.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_swap_flex_train.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_swap_flex_train.add_argument("--model_name", type=str, default="", help="Name of the SAEHD model")
    parser_swap_flex_train.add_argument("--iteration_goal", type=int, default=None, help="Iteration to reach " +
                                                                                         " for each face swapping " +
                                                                                         "operation")


def swap_extract_action_parser(subparsers):
    parser_extract = subparsers.add_parser("swap_extract", help="Extract faces from subjects")
    parser_extract.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_extract.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_extract.add_argument("--png_quality", type=int, help="Quality of the extracted images")


def dataframe_action_parser(subparsers):
    parser_dataframe = subparsers.add_parser("dataframe", help="Perform the creation of metadata.json file " +
                                                               "for original and merged video referencing.")
    parser_dataframe.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_dataframe.add_argument("--output_pickle", type=Path, default=None, help="Output path for the dataframe, " +
                                                                                   "by default : " +
                                                                                   " --subjects_dir/dataframe.pkl")


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


def swap_auto_action_parser(subparsers):
    parser_swap_auto = subparsers.add_parser("swap_auto", help="Face swapping")
    parser_swap_auto.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_swap_auto.add_argument("--model_dir", type=Path, help="Path to the models directory")
    parser_swap_auto.add_argument("--model_name", type=str, default="", help="Name of the SAEHD model")
    parser_swap_auto.add_argument("--iteration_goal", type=int, default=None, help="Iteration to reach " +
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


def raw_extract_action_parser(subparsers):
    parser_extract = subparsers.add_parser("extract", help="Extract faces from subjects")
    parser_extract.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_extract.add_argument("--dim_output_faces", type=int, help="Dimension of the output faces")
    parser_extract.add_argument("--png_quality", type=int, help="Quality of the extracted images")


def update_subject_workspace_action_parser(subparsers):
    parser_update_wrk = subparsers.add_parser("update_wrk", help="Update workspace")
    parser_update_wrk.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")


def clean_action_parser(subparsers):
    parser_clean = subparsers.add_parser("clean", help="Completely clean workspace, except if other option are given.")
    parser_clean.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
    parser_clean.add_argument("--redo_merged", action='store_true', help="Clean only the merged folder")
    parser_clean.add_argument("--redo_original", action='store_true', help="Clean only the frames and alignment folder")
    parser_clean.add_argument("--redo_face", action='store_true', help="Clean only the frames face folder")


def to_subject_action_parser(subparsers):
    parser_to_subject = subparsers.add_parser("to_subject", help="Convert videos to subjects")
    parser_to_subject.add_argument("--videos_dir", type=Path, help="Path to the videos directory")
    parser_to_subject.add_argument("--subjects_dir", type=Path, help="Path to the subjects directory")
