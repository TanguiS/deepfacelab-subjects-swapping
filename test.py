import json
from pathlib import Path

# List of elements with four values each
elements = [
    ['aagfhgtpmv.mp4', 'FAKE', 'train', 'vudstovrck.mp4'],
    ['aapnvogymq.mp4', 'FAKE', 'train', None]
]

# Create an empty dictionary
data = {}

# Iterate over the list of elements and add the values to the dictionary
for element in elements:
    key = element[0]
    value = {
        'label': element[1],
        'split': element[2],
        'original': element[3]
    }
    data[key] = value

# Write the dictionary as JSON to a file
filename = Path('C:\\WORK\\deepfacelab-subjects-swapping\\metadata.json')
filename.touch(exist_ok=True)
with open(filename, 'w') as file:
    json.dump(data, file)

print(f"JSON data has been written to '{filename}'")

