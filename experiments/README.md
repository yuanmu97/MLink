# Experiments

## cityscale-gtav

To evaluate the cross-domain adaptability of MLink, we use two video datasets, [Cityscale](https://www.cityscapes-dataset.com/) (street scenes in real cities) and [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) (street scenes in computer games), as different domains.

For Cityscale, we use the [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) (241MB) package. And for GTAV, we use the [Part1](https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip) data.

We consider two abstract models, one for car counting and another for person counting, and use annotations as simulated outputs.

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
So we are interested in predicting lowe-level outputs using higher level ones with MLink.

case#1: existence flag -> bounding box (`mlink_existence_to_bbox.py`). Not surprisingly, using only the flag that indicates whether cars exist cannot effectively predict cars' bounding boxes. The test bboxIoU is less than 4%.

case#2: bounding box -> pixel-level mask (`mlink_bbox_to_mask.py`). Using the 4-dim bounding box cannot effectively predict pixel-level mask (409440-dim vector). The test binaryIoU is less than 1%.

## sensitivity to output dimension


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
