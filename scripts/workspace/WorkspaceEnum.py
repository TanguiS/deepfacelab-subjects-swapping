import enum


class WorkspaceStr(enum.Enum):
    frames = "frames"
    aligned = "aligned"
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
    augmentation = "random_aligned_faces"
    f_aug = "fake"
    r_aug = "real"
    flex_model = "save_trained_model_on_each_subjects"
    model_on_sub = "model_on_"
    model_on_done_tag = ".done"
    model_on_retrain_tag = ".retrain"
