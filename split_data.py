import os
import shutil
import random


dataset_path = "data/raw"
output_path = "data/processed"
train_ratio = 0.8  # 80% for train, 20% for test


for split in ["train", "test"]:
    split_path = os.path.join(output_path, split)
    os.makedirs(split_path, exist_ok=True)

    for class_name in os.listdir(dataset_path):
        os.makedirs(os.path.join(split_path, class_name), exist_ok=True)

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)

    for i, img in enumerate(images):
        src = os.path.join(class_path, img)
        if i < train_size:
            dst = os.path.join(output_path, "train", class_name, img)
        else:
            dst = os.path.join(output_path, "test", class_name, img)
        shutil.copy2(src, dst)

print("✅ Dataset successfully split into train and test!")
