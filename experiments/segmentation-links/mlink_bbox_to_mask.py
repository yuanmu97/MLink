import sys
sys.path.append("../..")
from mlink.vec_to_vec import build_ModelLink_vec2vec, train_ModelLink_vec2vec
import numpy as np


input_len = 4
output_len = 409440 # 480*853

mlink = build_ModelLink_vec2vec(input_len, output_len, output_activation="sigmoid", hidden=0.0001)
print(mlink.summary())

bbox = np.load("outputs/boundingbox.npy")
vect = np.load("outputs/mask1d.npy")

# random shuffle
idx_list = np.arange(bbox.shape[0])
np.random.shuffle(idx_list)
vect = vect[idx_list]
bbox = bbox[idx_list]

# train-test split
train_n = int(0.8*bbox.shape[0])

res = train_ModelLink_vec2vec(mlink, bbox[:train_n], vect[:train_n], 
                              test_X=bbox[train_n:], test_Y=vect[train_n:],
                              learning_rate=0.01, batch_size=32, epochs=100,
                              log_dir="logs/", weight_path="weights/bbox2vect.h5",
                              loss="binary_crossentropy", metric="binary_iou")
