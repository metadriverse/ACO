import os

import zipfile
import numpy as np
import cv2

from torch.utils.data import Dataset
from PIL import Image

test = False
if test:
    ZIP = 'smallytb.zip'
    META = 'smeta.txt'
    PREFIX = 'smallytb'
else:
    ZIP = 'ytb.zip'
    META = 'meta.txt'
    PREFIX = 'newytb'

class ZipLoader(object):
    """Defines a class to load zip file.
    This is a static class, which is used to solve the problem that different
    data workers can not share the same memory.
    """
    files = dict()

    @staticmethod
    def get_zipfile(file_path):
        """Fetches a zip file."""
        zip_files = ZipLoader.files
        if file_path not in zip_files:
            zip_files[file_path] = zipfile.ZipFile(file_path, 'r')
        return zip_files[file_path]

    @staticmethod
    def get_image(file_path, image_path):
        """Decodes an image from a particular zip file."""
        zip_file = ZipLoader.get_zipfile(file_path)
        image_str = zip_file.read(image_path)
        image_np = np.frombuffer(image_str, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image

class YTBDataset(Dataset):
    def __init__(self, interval, data_path, phase, transform):
        super().__init__()

        self.zip_path = os.path.join(data_path, ZIP)
        self.zip_loader = ZipLoader()
        self.meta_path = os.path.join(data_path, META)
        self.meta_info = open(self.meta_path).readlines()   
        self.transform = transform
        self.phase = phase
        self.interval = interval
        total_len = len(self.meta_info) // self.interval
        self.eval_offset = int(total_len * 0.7) 
        assert phase in ['train', 'eval', 'all'], 'Not supported phase'

    def __len__(self):
        total_len = len(self.meta_info) // self.interval
        if self.phase == 'train':
            return int(total_len * 0.7) 
        elif self.phase == 'eval':
            return total_len - self.eval_offset 
        else:
            return total_len

    def __getitem__(self, index):
        if self.phase == 'eval':
            index += self.eval_offset
        index = index * self.interval
        info = self.meta_info[index]
        path, throttle, steering, speed = info.split()

        if index < 6510787:
            rindex = index
        else:
            rindex = index - 6510787

        path = path[:path.find('/')] + '/' + str(rindex) + path[-4:]

        throttle = float(throttle)
        steering = float(steering)
        speed = float(speed[:-3])
        img = self.zip_loader.get_image(self.zip_path, os.path.join(PREFIX, path))
        img = self.transform(Image.fromarray(img))
        return img, steering, throttle, speed

class LabelYTBDataset(YTBDataset):
    @staticmethod
    def _get_label(s, t):
        if s<0:
            s=0
        if s>1:
            s=1

        return s

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img, steering, throttle, speed = item
        label = self._get_label(steering, throttle)
        return img, label


if __name__ == '__main__':
    transform = None
    ytbd = YTBDataset(data_path='/data0/qh/data', phase='all', transform=transform, interval=1)
    print(len(ytbd))
