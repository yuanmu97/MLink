import tensorflow as tf 


def build_fusion(n_experts, output_len, 
                 output_activation='softmax'):
    inputs = list()
    for _ in range(n_experts):
        inputs.append(tf.keras.layers.Input(shape=(output_len, )))
    concated_input = tf.keras.layers.Concatenate()(inputs)
    dense1 = tf.keras.layers.Dense(output_len, activation=output_activation)(concated_input)

    fusion_model = tf.keras.Model(inputs=inputs, outputs=dense1)
    return fusion_model


def train_fusion(model, X_train, Y_train,
                 learning_rate=0.01,
                 loss="categorical_crossentropy",
                 metric="categorical_accuracy",
                 log_dir="logs/",
                 weight_path="weights/best.h5",
                 batch_size=1000,
                 epochs=200,
                 test_X=None, test_Y=None):
    metrics = [metric]

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )

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

    res = model.fit(
        x=X_train, y=Y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tb_callback, mcp_saver],
        validation_data=validation_data,
        verbose=0
    )

    model.save(weight_path+'.LAST')

    return res