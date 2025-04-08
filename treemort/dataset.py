import h5py
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DeadTreeDataset(Dataset):
    def __init__(self, hdf5_file_A, hdf5_file_B, keys_A, keys_B, crop_size=256, transform=None, crop=True):
        self.hdf5_file_A = hdf5_file_A
        self.hdf5_file_B = hdf5_file_B
        self.keys_A = keys_A
        self.keys_B = keys_B
        self.crop_size = crop_size
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return max(len(self.keys_A), len(self.keys_B))

    def __getitem__(self, idx):    
        idx_A = idx['idx_A']  
        idx_B = idx['idx_B']

        image_A, label_A, key_A, contains_dead_tree_A, filename_A = self._load_data_A(idx_A)
        image_B, label_B, key_B, contains_dead_tree_B, filename_B = self._load_data_B(idx_B)

        image_A, label_A = self._preprocess_image_and_label(image_A, label_A)
        image_B, label_B = self._preprocess_image_and_label(image_B, label_B)

        return {'A': image_A, 'label_A': label_A, 'key_A': key_A, 'contains_dead_tree_A': contains_dead_tree_A, 'filename_A': filename_A,
                'B': image_B, 'label_B': label_B, 'key_B': key_B, 'contains_dead_tree_B': contains_dead_tree_B, 'filename_B': filename_B}

    def _load_data_A(self, idx_A):
        key = self.keys_A[idx_A]

        with h5py.File(self.hdf5_file_A, "r") as hf:
            image = hf[key]['image'][()]
            label = hf[key]['label'][()]
            contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
            filename = hf[key].attrs.get("source_image", 0)

        return image, label, key, contains_dead_tree, filename
    
    def _load_data_B(self, idx_B):
        key = self.keys_B[idx_B]

        with h5py.File(self.hdf5_file_B, "r") as hf:
            image = hf[key]['image'][()]
            label = hf[key]['label'][()]
            contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
            filename = hf[key].attrs.get("source_image", 0)

        return image, label, key, contains_dead_tree, filename

    def _preprocess_image_and_label(self, image, label):
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        image = image / 255.0
        image = image * 2 - 1

        image = image.permute(2, 0, 1)  # Convert to (C, H, W) format
        label = label.unsqueeze(0)  # Convert to (1, H, W) format

        if self.crop:
            image, label = self._center_crop_or_pad(image, label, self.crop_size)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def _center_crop_or_pad(self, image, label, size=256):
        h, w = image.shape[1:]  # image is in (C, H, W) format

        pad_h, pad_w = max(size - h, 0), max(size - w, 0)

        if pad_h > 0 or pad_w > 0:
            image, label = self._pad_image_and_label(image, label, pad_h, pad_w)
        
        return self._crop_center(image, label, size)
        
    def _pad_image_and_label(self, image, label, pad_h, pad_w):
        image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0)
        label = F.pad(label, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0)

        return image, label
    
    def _crop_center(self, image, label, size):
        h, w = image.shape[1:]
        x, y = (w - size) // 2, (h - size) // 2

        return image[:, y:y + size, x:x + size], label[:, y:y + size, x:x + size]



### Image dataset for custom A to B tasks.
class ImageDataset(Dataset):
    def __init__(self, imagename_A, imagename_B, transform=None):
        self.imagename_A = imagename_A
        self.imagename_B = imagename_B
        self.transform = transform
        self.ToTensor = transforms.ToTensor()

    def __len__(self):
        return min(len(self.imagename_A), len(self.imagename_B))

    def __getitem__(self, idx):

        imagename_A = self.imagename_A[idx]
        imagename_B = self.imagename_B[idx]

        image_A = Image.open(imagename_A).convert('RGB')
        image_B = Image.open(imagename_B).convert('RGB')

        image_A = self._preprocess_image(image_A)
        image_B = self._preprocess_image(image_B)

        return {'A': image_A, 'imagename_A': imagename_A,
                'B': image_B, 'imagename_B': imagename_B}

    def _preprocess_image(self, image):

        image = self.ToTensor(image)

        image = image * 2 - 1

        if self.transform:
            image = self.transform(image)

        return image

    
    def _crop_center(self, image, size):
        h, w = image.shape[1:]
        x, y = (w - size) // 2, (h - size) // 2

        return image[:, y:y + size, x:x + size]