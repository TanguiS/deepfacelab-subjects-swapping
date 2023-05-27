import enum


class WorkspaceStr(enum.Enum):
    frames = "frames"
    aligned = frames + "/aligned"
    s_videos = "merged_videos"
    s_frames = "merged_frames"
    subject = "subject_"
    dst_video = "from_"
    videos = "output.*"
    tag = ".tag"
    pretrain = "pretrain_faces"
    mask = "mask"
    tmp_save = "tmp_save"
    metadata = "metadata.json"
    benchmark_csv = "benchmark.csv"
