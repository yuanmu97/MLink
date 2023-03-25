from losspred import create_mlink_with_loss_prediction
import numpy as np
import tensorflow as tf

def count_acc(y_pred, y_true):
    print(y_pred[:5], y_true[:5])
    correct = 0
    wrong = 0
    for y1, y2 in zip(y_pred, y_true):
        if abs(y1-y2[0]) < 0.5:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)

x_list = np.load('pred_vehicle.npy')
y_list = np.load('pred_person.npy')

split_n = 48
split_len = int(x_list.shape[0]/split_n)
k = int(split_len * 0.05)

LR = 0.01
BS = 32
EP = 50
mlink = create_mlink_with_loss_prediction(1, 1, 2, 0.0, 'linear', LR)
ckpt_saver = tf.keras.callbacks.ModelCheckpoint('weights/init.h5',
                                                monitor='loss',
                                                mode='min',
                                                save_best_only=True)

acc_list = []
for split_idx in range(split_n):
    s = split_idx*split_len
    pred = mlink.predict(x_list[s:s+split_len]) # shape=[SAMPLE_NUM, 2]
    loss_pred = pred[:, 1] # predicted losses

    topk_idx = np.argpartition(loss_pred, -k)[-k:]

    # train
    mlink.fit(x=x_list[s:s+split_len][topk_idx], y=y_list[s:s+split_len][topk_idx],
              batch_size=BS, epochs=EP, verbose=0)
    mlink.save(f"weights/split{split_idx:02d}.h5")

    mlink_pred = mlink.predict(x_list[s:s+split_len])[:, 0]
    acc = count_acc(mlink_pred, y_list[s:s+split_len])
    acc_list.append(acc)
    print(f"[Split{split_idx:02d}] Acc={acc}")

with open("res.txt", "w") as fout:
    fout.write("acc\n")
    for acc in acc_list:
        fout.write(f"{acc}\n")