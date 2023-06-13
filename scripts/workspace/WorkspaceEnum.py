import enum


class WorkspaceStr(enum.Enum):
    frames = "frames"
    aligned = "aligned"
    face = "face"
    merged_videos = "merged_videos"
    merged_frames = "merged_frames"
    subject = "subject_"
    merged_dst = "from_"
    videos = "output.*"
    tag = ".tag"
    pretrain = "pretrain_faces"
    mask = "mask"
    tmp_save = "tmp_save"
    benchmark_csv = "benchmark.csv"
    augmentation = "random_aligned_faces"
    fake_aug = "fake"
    real_aug = "real"
    flex_model = "save_trained_model_on_each_subjects"
    model_on_sub = "model_on_"
    model_on_done_tag = ".done"
    model_on_retrain_tag = ".retrain"
