import sys
sys.path.append("../..")
from mlink.vec_to_vec import build_ModelLink_vec2vec, train_ModelLink_vec2vec
import numpy as np


input_len = 4
output_len = 1

mlink = build_ModelLink_vec2vec(input_len, output_len, output_activation="sigmoid", hidden=32)
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

res = train_ModelLink_vec2vec(mlink, bbox[:train_n], exis[:train_n], 
                              test_X=bbox[train_n:], test_Y=exis[train_n:],
                              learning_rate=0.01, batch_size=32, epochs=100,
                              log_dir="logs/", weight_path="weights/bbox2exis.h5",
                              loss="binary_crossentropy", metric="accuracy")
