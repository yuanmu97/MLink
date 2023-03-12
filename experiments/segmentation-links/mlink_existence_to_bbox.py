import sys
sys.path.append("../..")
from mlink.vec_to_vec import build_ModelLink_vec2vec, train_ModelLink_vec2vec
import numpy as np


input_len = 1
output_len = 4

mlink = build_ModelLink_vec2vec(input_len, output_len, output_activation="linear")
print(mlink.summary())

exis = np.load("existence.npy")
bbox = np.load("boundingbox.npy")

# random shuffle
idx_list = np.arange(exis.shape[0])
np.random.shuffle(idx_list)
exis = exis[idx_list]
bbox = bbox[idx_list]

# train-test split
train_n = int(0.8*exis.shape[0])

res = train_ModelLink_vec2vec(mlink, exis[:train_n], bbox[:train_n], 
                              test_X=exis[train_n:], test_Y=bbox[train_n:],
                              learning_rate=0.01, batch_size=32, epochs=100,
                              log_dir="logs/", weight_path="weights/exis2bbox.h5",
                              loss="mse", metric="iou")
