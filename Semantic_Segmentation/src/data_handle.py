import h5py
from torch.utils.data import DataLoader, sampler, Dataset
import torchvision.transforms as T
import numpy as np
import torch

def seg2onehot(seg, n):
    """one-hot encode
    use one hot to encode the labels of pixels in a image

    Input:
    - seg: segmentation image, of shape(H, W, 1)
    - n: the number of labels

    Returns:
    - onehot: image after encoding, of shape(H, W, n)
    """
    H, W, _ = seg.shape
    onehot = np.zeros((H, W, n))
    mask = np.arange(seg.size) * n + seg.ravel()
    onehot.ravel()[mask - 1] = 1
    return onehot

def onehot2seg(onehot):
    """one-hot decode

    Input:
    - onehot: segmentation image after one-hot encodingt, of shape (H, W, n)
    
    Returns:
    - seg: decoding results, of shape (H, W, 1)
    """
    seg = np.argmax(onehot, axis=2)
    return seg

def seg2rgb(seg, color_codes):
    """transform segmentation image into rgb image

    Args:
        seg (ndarray): segmentation, of shape (H, W)
        color_codes (ndarray): A list of rgb color codes corresponding to classes

    Returns:
        ndarray: a rgb image, of shape (H, W, 3)
    """    
    H, W = seg.shape
    rgb = np.zeros((H, W, 3))
    for i in range(color_codes.shape[0]):
        rgb[seg==i] = color_codes[i]
    return rgb

class data_loader(Dataset):
    def __init__(self, path, transform=None):        
        """
        Input:
        - dataset: A tuple (rgb, seg)
        - num_class: the number of labels
        - transform: 
        """
        # super(data_loader, self).__init__()
        file = h5py.File(path,"r")
        self.transform = transform
        self.x = file['rgb'][:]
        self.y = file['seg'][:]
        self.length = self.x.shape[0]
        self.color_codes = file['color_codes'][:]
        self.num_class = self.color_codes.shape[0]

    def color_codes(self):
        return self.color_codes
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.x[index]  # of shape (H, W, 3)
        seg = self.y[index]  # of shape (H, W, 1)
        
        # seg = seg2onehot(seg, self.num_class) # (H, W, n)
        seg = torch.as_tensor(seg.squeeze()) # (H, W)
        
        if self.transform is not None:
            img = self.transform(img)

        return img.float(), seg.type(torch.LongTensor)