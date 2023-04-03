import numpy as np
from spherical_distortion import *
from glob import glob
from tqdm import tqdm
import os
import random
from joblib import Parallel, delayed

random.seed(1)
np.random.seed(1)

DATA_DIR = "./calibration_dataset/"

test_images_len = 2000 
val_images_len = 5000

all_images = [os.path.splitext(os.path.basename(x))[0] for x in glob(DATA_DIR + "*.jpg", recursive=False)]
all_panos = list(set(map(lambda x: x[:-3], all_images)))
train_images_len = len(all_images) - (test_images_len+val_images_len)

np.random.shuffle(all_panos)
trainval_panos = all_panos[0:int((val_images_len+train_images_len)/len(all_images)*len(all_panos))]
test_panos = all_panos[-int(test_images_len/len(all_images)*len(all_panos)):]
# test = np.random.choice(all_panoramas, size=int(test_ratio*len(all_panos)), replace=False)
print(f"Total number of panos available: {len(all_panos)}. Trainval_panos len: {len(trainval_panos)}. Test panos len: {len(test_panos)}")

# TODO: make this more efficient
trainval_images = []
test_images = []
train_images = []
val_images = []

for image in tqdm(all_images):
    for pano in trainval_panos:
        if pano in image:
            trainval_images.append(image)
            break
    for pano in test_panos:
        if pano in image:
            test_images.append(image)
            break
            
np.random.shuffle(trainval_images)
np.random.shuffle(test_images)

train_images = trainval_images[0:-val_images_len]
val_images = trainval_images[-val_images_len:]

with open("split/train_data.txt", "w+") as train_file:
    train_file.write("\n".join(train_images))
with open("split/val_data.txt", "w+") as val_file:
    val_file.write("\n".join(val_images))
with open("split/test_data.txt", "w+") as test_file:
    test_file.write("\n".join(test_images))