import random
import torch

from torchvision import transforms

def random_flip_wlabel(image, label):
    if random.random() > 0.5:
        image = torch.flip(image, [2])  # Horizontal flip
        label = torch.flip(label, [2])
    if random.random() > 0.5:
        image = torch.flip(image, [1])  # Vertical flip
        label = torch.flip(label, [1])
    return image, label

def random_rotation_wlabel(image, label):
    k = random.randint(0, 3)
    image = torch.rot90(image, k, [1, 2])
    label = torch.rot90(label, k, [1, 2])
    return image, label

def random_flip(image):
    if random.random() > 0.5:
        image = torch.flip(image, [2])  # Horizontal flip
    #if random.random() > 0.5:
    #    image = torch.flip(image, [1])  # Vertical flip
    return image

def random_rotation(image):
    k = random.randint(0, 3)
    image = torch.rot90(image, k, [1, 2])
    return image

def random_brightness(image):
    factor = 1.0 + random.uniform(-0.2, 0.2)
    image = torch.clamp(image * factor, 0, 1)
    return image

def random_contrast(image):
    factor = 1.0 + random.uniform(-0.2, 0.2)
    mean = torch.mean(image, dim=(1, 2), keepdim=True)
    image = torch.clamp((image - mean) * factor + mean, 0, 1)
    return image

def random_multiplicative_noise(image):
    noise = torch.rand_like(image) * 0.2 + 0.9
    image = torch.clamp(image * noise, 0, 1)
    return image

def random_gamma(image):
    gamma = random.uniform(0.8, 1.2)
    image = torch.clamp(image**gamma, 0, 1)
    return image

class Augmentations_wlabel:
    def __call__(self, image, label):
        # Scale between 0 and 1.
        image = image + 1
        image = image/2
        image, label = random_flip_wlabel(image, label)
        image, label = random_rotation_wlabel(image, label)
        image = random_brightness(image)
        image = random_contrast(image)
        image = random_multiplicative_noise(image)
        image = random_gamma(image)
        # Scale back to between -1 and 1.
        image = image * 2 - 1
        return image, label

class Augmentations:
    def __init__(self, train_load_size=286, train_crop_size=256, augment_mode='full'):
        
        self.augment_mode = augment_mode

        osize = [train_load_size, train_load_size]
        self.resize = transforms.Resize(osize,
                                        interpolation=transforms.functional.InterpolationMode.BICUBIC)
        self.crop = transforms.RandomCrop(train_crop_size)

    def __call__(self, image):
        # Scale between 0 and 1.
        image = image + 1
        image = image/2
        if self.augment_mode == 'full':
            image = random_flip(image)
            #image = random_rotation(image)
            image = random_brightness(image)
            image = random_contrast(image)
            #image = random_multiplicative_noise(image)
            image = random_gamma(image)
        else:
            image = self.resize(image)
            image = self.crop(image)
        # Scale back to between -1 and 1.
        image = image * 2 - 1
        return image