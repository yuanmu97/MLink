import numpy as np
import os
from tqdm import tqdm

data_dir = "deeplabv3_predictions/"
data_num = len(os.listdir(data_dir))

bbox_list = []
exis_list = []

for i in tqdm(range(data_num)):
    d = np.load(os.path.join(data_dir, f"bs12_batch{i:04d}.npy"))
    for tmp in d:
        car_seg = (tmp == 7).astype(int)
        
        if np.sum(car_seg) == 0:
            # no cars
            bbox = [0., 0., 0., 0.]
            existence = [0.]
        else:
            y, x = np.where(car_seg == 1)
            bbox = [min(x), min(y), max(x)-min(x), max(y)-min(y)]
            existence = [1.]
        
        bbox_list.append(bbox)
        exis_list.append(existence)

bbox_list = np.array(bbox_list)
exis_list = np.array(exis_list)

print(bbox_list.shape, exis_list.shape)
np.save("boundingbox.npy", bbox_list)
np.save("existence.npy", exis_list)