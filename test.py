import json
from pathlib import Path

fake = Path("path/to/subjects/subjects_x/merged_videos/videos_from_subject_y.mov")

tmp = fake.parent
fake = str(tmp.name) + "/" + str(fake.name)

print(fake)