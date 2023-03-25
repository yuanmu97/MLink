# Experiments

## cityscapes-gtav

To evaluate the cross-domain adaptability of MLink, we use two video datasets, [Cityscale](https://www.cityscapes-dataset.com/) (street scenes in real cities) and [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) (street scenes in computer games), as different domains.

We deploy [DeepLabv3Plus](https://github.com/VainF/DeepLabV3Plus-Pytorch) pretrained on Cityscapes and run inference on GTAV images. We save segmentation prediction as .npy files.
```python
'''
DeepLabV3Plus-Pytorch/predict.py
'''
# ...
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW

            # colorized_preds = decode_fn(pred).astype('uint8')
            # colorized_preds = Image.fromarray(colorized_preds)
            # if opts.save_val_results_to:
            #     colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+opts.model+'.png'))
    
            np.save(os.path.join(opts.save_npy, img_name+".npy"), pred)
# ...
```
Then we train a neural network that maps the segmentation prediction to groundtruth masks.
See `train_seg_mlink.py`.
We use Conv2D to extract features and use Conv2DTranspose for upsampling.
Experimental results show that mlink cannot be effectively learned for such large (1052*1912) output spaces.


<!-- For Cityscapes, we use the [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) (241MB) package. And for GTAV, we use the [Part1](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip) data.

We consider two abstract models, one for car counting and another for person counting, and use annotations as simulated outputs. -->

## segmentation links

For more comprehensive evaluations on real-world video analytics, we use [DeepLabV3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) model for semantic segmentation on our collected traffic videos.

DeepLabV3 predicts pixel-level segmentation of 21 types of objects:
```python
labelnames = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

Run DeepLabV3 on video frames:
```bash
python run_deeplab3.py
# segmentation predictions are saved in ./deeplabv3_predictions/
```

Next, generate three levels of outputs related to "car" labels:

1. pixel-level mask
2. bounding box
3. existence flag

Run `python generate_outputs.py`.

Obviously, the higher level outputs can be easily computed using lower level ones, e.g., `[min(maskx), min(masky), max(maskx), max(masky)]=bbox` and `(len(bbox_list)!=0)=existence`.
Our experiments verify this intuition: `mlink_bbox_to_existence.py` results in 100% val accuracy.
However, since we use flattened mask as the input, `mlink_mask_to_bbox.py` only achieves 16.5% IoU.
And we are interested in predicting lowe-level outputs using higher level ones with MLink.

case#1: existence flag -> bounding box (`mlink_existence_to_bbox.py`). Not surprisingly, using only the flag that indicates whether cars exist cannot effectively predict cars' bounding boxes. The test bboxIoU is less than 4%.

case#2: bounding box -> pixel-level mask (`mlink_bbox_to_mask.py`). Using the 4-dim bounding box cannot effectively predict pixel-level mask (409440-dim vector). The test binaryIoU is less than 1%.


## communication-overhead

Using [TensorFlow Federated](https://github.com/tensorflow/federated), multiple edges can train global model links via communication.
We test communication overhead of aggregating model links of action classification models.
The action classifier is pre-trained with [Action-Net](https://github.com/OlafenwaMoses/Action-Net) which contains 16 different actions.
To better explore the general cases, we generate random 16-dim outputs with domain shifts using `generate_domains.ipynb`:

```python
def generate_pred(x, sigma=0.5, label_n=16):
    preds = []
    for t in x:
        bins = np.arange(1, label_n+1)
        p = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - t)**2 / (2 * sigma**2))
        preds.append(p)
    preds = np.stack(preds)
    return preds

output_len = 16
locs = [4, 8, 12]
shifts = [5, -3, 1]
std = 2
data_num = 1000

x1 = generate_pred(np.random.normal(locs[0], std, data_num))
y1 = generate_pred(np.random.normal(locs[0]+shifts[0], std, data_num))
print("Domain#1:", x1.shape, locs[0], y1.shape, locs[0]+shifts[0])

x2 = generate_pred(np.random.normal(locs[1], std, data_num))
y2 = generate_pred(np.random.normal(locs[1]+shifts[1], std, data_num))
print("Domain#2:", x2.shape, locs[1], y2.shape, locs[1]+shifts[1])

x3 = generate_pred(np.random.normal(locs[2], std, data_num))
y3 = generate_pred(np.random.normal(locs[2]+shifts[2], std, data_num))
print("Domain#3:", x3.shape, locs[2], y3.shape, locs[2]+shifts[2])
```
Next, we train the global model link using three clients with `mlink.federated` module.

```bash
python run_federated.py
...
SERVER_UPDATED_WEIGHTS
 [<tf.Tensor 'sub_8:0' shape=(16, 32) dtype=float32>, <tf.Tensor 'sub_9:0' shape=(32,) dtype=float32>, <tf.Tensor 'sub_10:0' shape=(32, 16) dtype=float32>, <tf.Tensor 'sub_11:0' shape=(16,) dtype=float32>]
SIZE=88 bytes
CLIENT_UPDATE
 [<tf.Tensor 'Sub:0' shape=(16, 32) dtype=float32>, <tf.Tensor 'Sub_1:0' shape=(32,) dtype=float32>, <tf.Tensor 'Sub_2:0' shape=(32, 16) dtype=float32>, <tf.Tensor 'Sub_3:0' shape=(16,) dtype=float32>]
SIZE=88 bytes
...
```
So for the 3-client case, the overall communication cost is 88\*3 (server broadcast) + 88\*3 (client update) = 528 bytes per sample.
We set the batch size as 32, which translates into communication cost of 16.5 KB per training step.

## online adaptation

Besides uncertainty-based active policy, we adopt a loss prediction-based method ([Learning Loss for Active Learning, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.html)).
The key idea is adding a neural network that takes the intermediate activation as input and predict the sample loss.
We implement the joint loss as follows: 

```python
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
```

And now model link's architecture has two output branch, one for original task and another for loss prediction:

```python
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
```

For training and evaluation, see `online_train.py`.