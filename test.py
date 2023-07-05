import json
from pathlib import Path

import pandas as pd

from scripts.extract.face.yunet_scripts import load_face_features_extraction_model

model_p = Path("/media/tangui/CA1EF5E61EF5CC07/ubuntu_drive/DeepFaceLab_data/my_data/model_on_subjects/model")

test = load_face_features_extraction_model(model_p)


a = 0