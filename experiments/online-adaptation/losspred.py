"""
losspred.py

yuanmu
"""
import tensorflow as tf
from tensorflow.keras import layers


def joint_loss(y_true, y_pred):
    mlink_pred = y_pred[:, 0:1]
    loss_pred = y_pred[:, 1:2]

    # mlink loss
    mlink_loss = tf.keras.losses.mae(y_true, mlink_pred)

    # loss-prediction loss
    def lp_loss(loss_true, loss_pred, margin=1.0):
        loss_pred = (loss_pred - tf.reverse(loss_pred, axis=[0]))[:len(loss_pred)//2]
        loss_true = (loss_true - tf.reverse(loss_true, axis=[0]))[:len(loss_true)//2]
        loss_true_stop = tf.stop_gradient(loss_true)
        one = 2 * tf.math.sign(tf.clip_by_value(loss_true_stop, clip_value_min=0, clip_value_max=tf.float32.max)) - 1
        tmp = margin - one * loss_pred
        tmp = tf.clip_by_value(tmp, clip_value_min=0, clip_value_max=tf.float32.max)
        res = tf.reduce_mean(tmp)
        return res
    
    loss_pred_loss = lp_loss(mlink_loss, loss_pred)
    res = mlink_loss + loss_pred_loss
    return res


def create_mlink_with_loss_prediction(input_len, output_len, 
                                      hidden=2, dropout=0.5, output_activation='softmax',
                                      learning_rate=0.01):
    # base model link
    inp = layers.Input(shape=(input_len, ))
    fc1 = layers.Dense(int(output_len * hidden))(inp)
    dropout1 = layers.Dropout(dropout)(fc1)
    relu1 = layers.Activation('relu')(dropout1)
    fc2 = layers.Dense(output_len)(relu1)
    out = layers.Activation(output_activation, name="mlink_out")(fc2)

    # loss prediction
    loss_out = layers.Dense(1, name="loss_out")(fc2)

    # concatenated output
    concate_out = layers.Concatenate()([out, loss_out])

    mlink = tf.keras.Model(inputs=inp, outputs=concate_out)

    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    mlink.compile(optimizer=opt, loss=joint_loss, metrics="mae")
    return mlink


if __name__ == "__main__":
    m = create_mlink_with_loss_prediction(1, 1, 2, 0.0, "linear")
    print(m.summary())
