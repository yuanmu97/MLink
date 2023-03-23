import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image

def build_mlink(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    
    # encoder
    conv1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inp)
    pool1 = layers.MaxPool2D((2, 2))(conv1)
    conv2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
    pool2 = layers.MaxPool2D((2,2))(conv2)
    conv3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(pool2)
    
    # decoder
    up1 = layers.Conv2DTranspose(32, (3, 3), strides=(2,2), activation="relu", padding="same")(conv3)
    conv4 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(up1)
    up2 = layers.Conv2DTranspose(16, (3, 3), strides=(2,2), activation="relu", padding="same")(conv4)
    conv5 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(up2)
    
    out = layers.Conv2D(num_classes, 1, activation="softmax")(conv5)
    
    return tf.keras.Model(inputs=inp, outputs=out)

def load_dataset(num=2500, only_input=False):
    assert num <= 2500
    filenames = [f"{i+1:05d}" for i in range(num)]
    name_ds = tf.data.Dataset.from_tensor_slices(filenames)
    
    def load_npy(name):
        name = bytes.decode(name.numpy())
        path = os.path.join("DeepLabV3Plus-Pytorch/test_results", f"{name}.npy")
        inp = np.load(path)[:,:1912] # (1052, 1914) -> (1052, 1912)
        inp = np.expand_dims(inp, -1) # (1052, 1912) -> (1052, 1912, 1)
        return inp
    
    input_ds = name_ds.map(lambda x: tf.py_function(load_npy, [x], [tf.float32]))
    
    if only_input:
        return input_ds
    
    def load_label(name):
        name = bytes.decode(name.numpy())
        img = Image.open(f"01_labels/{name}.png")
        img = np.array(img)[:,:1912]
        
        id_to_trainid = np.array([
            19, 19, 19, 19, 19, 19, 19,
            0, 1,
            19, 19,
            2, 3, 4,
            19, 19, 19,
            5,
            19,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
            19, 19,
            16, 17, 18,
            19
        ])
        img = id_to_trainid[img]
        return img
    
    label_ds = name_ds.map(lambda x: tf.py_function(load_label, [x], [tf.float32]))
    return tf.data.Dataset.zip((input_ds, label_ds))


BS = 1
EP = 100
LR = 0.01

m = build_mlink(input_shape=(1052, 1912, 1), num_classes=19+1)

print(m.summary())

opt = tf.keras.optimizers.RMSprop(learning_rate=LR)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
m.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])
ckpt_saver = tf.keras.callbacks.ModelCheckpoint("weights/test.best.h5", monitor="sparse_categorical_accuracy", mode="max", save_best_only=True)

d = load_dataset()
d = d.batch(BS)

m.fit(d, batch_size=BS, epochs=EP, callbacks=[ckpt_saver])
m.save(f"weights/test.ep{EP}.h5")