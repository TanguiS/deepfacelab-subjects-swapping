import shutil
from pathlib import Path
from typing import List

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from scripts.extract import util
from scripts.extract.face import face_extract
from scripts.extract.face.yunet import YuNet
from scripts.workspace.WorkspaceEnum import WorkspaceStr, video_extensions


def load_asset(subjects_dir):
    rd_data_aug = subjects_dir.joinpath(WorkspaceStr.augmentation.value)
    df_history_path = rd_data_aug.joinpath("history.pkl")
    if df_history_path.exists():
        df_history = pd.read_pickle(df_history_path)
    else:
        df_history = pd.DataFrame()
        df_history['processed'] = ''
    return df_history, df_history_path, rd_data_aug


def save_asset(df_history, df_history_path):
    print(f"Saving dataframe to {df_history_path}...")
    df_history.to_pickle(str(df_history_path))


def video_to_extract_frames(data_folder: Path, df_history: DataFrame) -> List[Path]:
    videos = []
    for sub_folder_name in (WorkspaceStr.real_aug.value, WorkspaceStr.fake_aug.value):
        sub_folder = data_folder.joinpath(sub_folder_name)

        for video in sub_folder.joinpath(WorkspaceStr.videos.value).iterdir():
            if not video.exists() and not video.is_file():
                continue
            if video.suffix.lower() not in video_extensions:
                continue
            if video.name in df_history['processed'].values:
                continue
            videos.append(video)
    return videos


def frame_to_extract_face(data_folder: Path, df_history: DataFrame) -> List[Path]:
    frames = []

    for sub_folder_name in (WorkspaceStr.real_aug.value, WorkspaceStr.fake_aug.value):
        sub_folder = data_folder.joinpath(sub_folder_name)
        for frame in sub_folder.joinpath(WorkspaceStr.frames.value).iterdir():
            if not frame.exists() and not frame.is_file():
                continue
            if frame.name in df_history['processed'].values:
                continue
            frames.append(frame)
    return frames


def launch_video_extract_frames(subjects_dir: Path) -> None:
    df_history, df_history_path, rd_data_aug = load_asset(subjects_dir)

    videos = video_to_extract_frames(rd_data_aug, df_history)

    for video in tqdm(videos, total=len(videos), desc="Extracting frames from videos...", unit=" video"):
        tmp_dst_dir = video.parent.joinpath(WorkspaceStr.tmp_save.value)
        tmp_dst_dir.mkdir(exist_ok=True)
        dst_dir = video.parent.parent.joinpath(WorkspaceStr.frames.value)
        util.video_to_frames(video, tmp_dst_dir, video.stem)
        for frame in tmp_dst_dir.iterdir():
            shutil.move(str(frame), str(dst_dir))
        shutil.rmtree(str(tmp_dst_dir))
        df_history = df_history.append([{'processed': video.name}], ignore_index=True)

    save_asset(df_history, df_history_path)


def launch_face_extract_frames(subjects_dir: Path, face_detector_model: YuNet, max_shape: int = 112) -> None:
    df_history, df_history_path, rd_data_aug = load_asset(subjects_dir)

    frames = frame_to_extract_face(rd_data_aug, df_history)
    for frame in tqdm(frames, total=len(frames), desc="Extracting face from frame...", unit=" frame"):
        dst_dir = frame.parent.parent.joinpath(WorkspaceStr.face.value)
        extracted_face = face_extract.extract_face(frame, face_detector_model, max_shape)
        face_extract.save_extracted_face(extracted_face, dst_dir, f"{max_shape}_{frame.name}")
        df_history = df_history.append([{'processed': frame.name}], ignore_index=True)

    save_asset(df_history, df_history_path)
