import sys
sys.path.append("../..")
from mlink.federated import run_federated_mlink_vec2vec_v2
import numpy as np
import tensorflow as tf


x1 = np.load("x1.npy")
y1 = np.load("y1.npy")
d1 = tf.data.Dataset.from_tensor_slices((x1, y1)).batch(32)

x2 = np.load("x2.npy")
y2 = np.load("y2.npy")
d2 = tf.data.Dataset.from_tensor_slices((x2, y2)).batch(32)

x3 = np.load("x3.npy")
y3 = np.load("y3.npy")
d3 = tf.data.Dataset.from_tensor_slices((x3, y3)).batch(32)

fed_d = [d1, d2, d3]

run_federated_mlink_vec2vec_v2(16, 16, fed_d, epochs=5)