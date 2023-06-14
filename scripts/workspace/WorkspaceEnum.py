import enum


class WorkspaceStr(enum.Enum):
    frames = "frames"
    aligned = "aligned"
    face = "face"
    merged_videos = "merged_videos"
    merged_frames = "merged_frames"
    subject = "subject_"
    merged_dst = "from_"
    videos_pattern = "output.*"
    videos = "videos"
    tag = ".tag"
    pretrain = "pretrain_faces"
    mask = "mask"
    tmp_save = "tmp_save"
    benchmark_csv = "benchmark.csv"
    augmentation = "random_data_augmentation"
    fake_aug = "fake"
    real_aug = "real"
    flex_model = "save_trained_model_on_each_subjects"
    model_on_sub = "model_on_"
    model_on_done_tag = ".done"
    model_on_retrain_tag = ".retrain"


video_extensions = {
    ".3g2", ".3gp", ".amv", ".asf", ".avi", ".drc", ".flv", ".f4v", ".f4p", ".f4a",
    ".f4b", ".gif", ".gifv", ".m4v", ".mkv", ".mng", ".mov", ".mp2", ".mp4", ".m4p",
    ".m4v", ".mpg", ".mp2", ".mpeg", ".mpe", ".mpv", ".mpg", ".mpeg", ".m2v", ".m4v",
    ".mxf", ".nsv", ".ogg", ".ogv", ".qt", ".rm", ".rmvb", ".roq", ".svi", ".vob",
    ".webm", ".wmv", ".yuv"
}
