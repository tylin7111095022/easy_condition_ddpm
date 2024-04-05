import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return sorted(images)

def pil_loader(path):
    return Image.open(path).convert('RGB')

class AICUPDataset(data.Dataset):
    def __init__(self, data_root, condition_root, data_len=-1, image_size=[120, 214], loader=pil_loader):
        imgs = make_dataset(data_root)
        conditions = make_dataset(condition_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
            self.conditions = conditions[:int(data_len)]
        else:
            self.imgs = imgs
            self.conditions = conditions
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        gtpath = self.imgs[index]
        condpath = self.conditions[index]
        assert os.path.basename(gtpath).split(".")[0] == os.path.basename(condpath).split(".")[0], "img dismatch with condition_imgs."
        img = self.tfs(self.loader(gtpath))
        cond_image = self.tfs(self.loader(condpath))
        # ret = {}
        # ret['gt_image'] = img
        # ret['cond_image'] = cond_image
        return img, cond_image

    def __len__(self):
        return len(self.imgs)