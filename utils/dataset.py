import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    image_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                image_paths.append(path)                
    assert len(image_paths) > 0, '[*]The number of input images should not zero'
    return image_paths
    

def select_dataset(dir, max_num, rand_select, rand_seed):
    image_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                image_paths.append(path)
    print("[*]Loaded {} original images, selected {} images for watermarking".format(len(image_paths), max_num))
    assert len(image_paths) >= max_num, '[*]Total loaded images number should bigger than selected images'
    if rand_select=='Yes':
        np.random.seed(rand_seed)
        selected_imgpaths = np.random.choice(image_paths, max_num)
    else:
        selected_imgpaths = image_paths[:max_num]
    return selected_imgpaths


def label_dataset(dir, label):
    data_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                data_paths.append((path, label))                
    assert len(data_paths) > 0, '[*]The number of input images should not zero'
    return data_paths



class TrainingDataset(Dataset):
    def __init__(self, root):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(make_dataset(root))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transforms(image)
        except:
            self.__getitem__(index + 1)



class InjectionDataset(Dataset):
    def __init__(self, root, max_num, rand_select, rand_seed):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(select_dataset(root, max_num, rand_select, rand_seed))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image)



class AnalysisDataset(Dataset):
    def __init__(self, root, perturbation):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(make_dataset(root))
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        if perturbation == 'Yes':
            print("[*]Loading image with perturbation")
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.6, 1.0)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)
            ])
        else:
            print("[*]Loading image without perturbation")
            self.transforms = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor()
            ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image)



class EvaluationDataset(Dataset):
    def __init__(self, pos_root, neg_root):
        self.data_list = []

        print("[*]Loading positive images from {}".format(pos_root))
        pos_paths = sorted(label_dataset(pos_root, label=1))
        self.data_list.extend(pos_paths)

        print("[*]Loading negative images from {}".format(neg_root))
        neg_paths = sorted(label_dataset(neg_root, label=0))
        self.data_list.extend(neg_paths)

        print('[*]{} images have been loaded, and there are {} positive images and {} negative images'.format(len(self.data_list), len(pos_paths), len(neg_paths)))

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        return self.transforms(image), label