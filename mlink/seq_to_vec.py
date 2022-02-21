import tensorflow as tf 
import numpy as np 
from .utils import BBoxIoU


def build_ModelLink_seq2vec(input_dim, output_len, 
                            hidden=2,
                            output_activation='softmax'):
    """build_ModelLink_seq2vec

    Args:
        input_dim (int): input vocabulary size
        output_len (int): length of outputs
        output_activation (str): softmax, sigmoid, linear

    Returns:
        mlink (keras Model): seq2vec ModelLink
    """
    mlink = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim, hidden*output_len),
        tf.keras.layers.LSTM(hidden*output_len),
        tf.keras.layers.Dense(output_len),
        tf.keras.layers.Activation(output_activation)
    ])
    return mlink


def train_ModelLink_seq2vec(mlink, X, Y,
                            learning_rate=0.01,
                            batch_size=32,
                            epochs=100,
                            log_dir="logs/",
                            weight_path="weights/ckpt.h5",
                            test_X=None, test_Y=None,
                            loss='categorical_crossentropy',
                            metric='accuracy'):
    """train_ModelLink_seq2vec
    
    Args:
        mlink (keras Model): the seq2vec ModelLink
        X, Y (array): fixed-length training samples
        learning_rate (float)
        batch_size (int)
        epochs (int)
        log_dir (str): tensorboard log_dir
        weight_path (str): path/to/ckpt.h5
        test_X, test_Y (array): testing samples
        loss (str): loss function name (categorical_crossentropy, mae, mse)
        metric (str): metric name (accuracy, iou, mea, mse)
    
    Returns:
        res (History): training & validation history object
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    if metric == "iou":
        metrics = [BBoxIoU()]
    else:
        metrics = [metric]

    mlink.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    if (test_X is not None) and (test_Y is not None):
        validation_data = (test_X, test_Y)
    else:
        validation_data = None
    
    # train
    res = mlink.fit(x=X, y=Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[tb_callback],
                    validation_data=validation_data,
                    verbose=0)
    
    # checkpoint
    mlink.save(weight_path)

    return res

