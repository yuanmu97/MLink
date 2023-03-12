import tensorflow as tf 
from .utils import BBoxIoU, BinaryIoU


def build_ModelLink_vec2vec(input_len, output_len,
                            hidden=2,
                            dropout=0.5,
                            output_activation='softmax'):
    """build_ModelLink_vec2vec
    build the ModelLink with fixed-length input and output

    Args:
        input_len, output_len (int): length of input, output
        hidden (int): neuron # in hidden dense layer (output_len * hidden)
        dropout (flot): dropout ratio
        output_activation: softmax, sigmoid, linear

    Returns:
        mlink (keras Model): ModelLink model
    """
    mlink = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_len,)),
        tf.keras.layers.Dense(int(output_len*hidden)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(output_len, activation=output_activation),
    ])
    return mlink


def load_ModelLink_vec2vec(checkpoint):
    return tf.keras.models.load_model(checkpoint, compile=False)


def train_ModelLink_vec2vec(mlink, X, Y,
                            learning_rate=0.01,
                            batch_size=32,
                            epochs=100,
                            log_dir="logs/",
                            weight_path="weights/ckpt.h5",
                            test_X=None, test_Y=None,
                            loss='categorical_crossentropy',
                            metric='accuracy'):
    """train_ModelLink_vec2vec
    
    Args:
        mlink (keras Model): the vec2vec ModelLink
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
    elif metric == "binary_iou":
        metrics = [BinaryIoU()]
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
    
    if metric == "categorical_accuracy":
        mcp_saver = tf.keras.callbacks.ModelCheckpoint(weight_path+'.BEST', 
            monitor='val_categorical_accuracy', 
            mode='max',
            save_best_only=True
        )
    else:
        mcp_saver = tf.keras.callbacks.ModelCheckpoint(weight_path+'.BEST', 
            monitor='val_loss', 
            mode='min',
            save_best_only=True
        )

    # train
    res = mlink.fit(x=X, y=Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[tb_callback, mcp_saver],
                    validation_data=validation_data)
    
    # checkpoint
    mlink.save(weight_path+'.LAST')

    return res