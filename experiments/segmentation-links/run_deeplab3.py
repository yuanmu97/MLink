import torch
from torch.utils.data import Dataset, DataLoader 
from PIL import Image
from torchvision import transforms
import numpy as np


class CityDataset(Dataset):
    def __init__(self, list_file):
        super(CityDataset, self).__init__()

        self.path_list = []
        with open(list_file, "r") as fin:
            for line in fin.readlines():
                self.path_list.append(line.strip())
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img = Image.open(self.path_list[index]).convert("RGB")
        img = self.preprocess(img)
        return img
    
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()
model.to('cuda')

d = CityDataset("list.txt")
d_loader = DataLoader(d, batch_size=12)

batch_n = len(d)/12

with torch.no_grad():
    for idx, x in enumerate(d_loader):
        x = x.cuda()
        y = model(x)['out'].argmax(1).cpu().numpy()
        np.save(f"deeplabv3_predictions/bs12_batch{idx:04d}.npy", y)
        if idx % 10 == 0:
            print(f"Batch {idx}/{batch_n}")
print("DONE.")