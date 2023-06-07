import json
from pathlib import Path

import pandas as pd

dataset = Path('D:\storage-photos\subjects\output.pkl')
df = pd.read_pickle(dataset)

a = 0