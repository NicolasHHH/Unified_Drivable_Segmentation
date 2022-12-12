import os
from pathlib import Path
from tqdm import tqdm
import shutil
import json

json_count = 0
car_count = 0
other_count = 0
person_count = 0

label_root = Path("/media/hty/T7/bdd/det_annotations/100k/val")
label_list = list(label_root.iterdir())  # json files
dst_root = Path("/media/hty/T7/bdd/det_annotations/100k/val_person")

# for label in tqdm(label_list[0:30000]):
#     label_path = str(label)
#     with open(label_path, 'r') as f:  # read label file
#         label = json.load(f)
#     data = label['frames'][0]['objects']
#     json_count += 1
#     for obj in data:
#         if obj["category"] == "car":
#             car_count += 1
#         elif obj["category"] == "person":
#             person_count += 1
#         else:
#             other_count += 1
# print("cars: ", car_count)
# print("people: ", person_count)
# print("others: ", other_count)


for label in tqdm(label_list):
    label_path = str(label)
    with open(label_path, 'r') as f:  # read label file
        label = json.load(f)
    data = label['frames'][0]['objects']
    for obj in data:
        if obj["category"] == "person":
            shutil.copyfile(label_path, label_path.replace("val","val_person"))
        break