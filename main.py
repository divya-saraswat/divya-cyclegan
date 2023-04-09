import random
import os
import numpy as np
import torch
from dataset import ImageDataset, DehazeDataset, HazeDataset, load_data
from utils import show_example
from train import do_train


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(1003)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dehaze_path = "data/normal/normal_images"
    haze_path = "data/hazy/hazy_images"
    size = (64, 64)
    data = ImageDataset(dehaze_path, haze_path, size)
    dehaze_data = DehazeDataset(dehaze_path, size)
    haze_data = HazeDataset(haze_path, size)
    data_loaded = load_data(data)
    dehaze_data_loaded = load_data(dehaze_data)
    haze_data_loaded = load_data(haze_data)
    show_example(data_loaded)
    pretrained_path = "./cycleGAN_10000.pth"
    do_train(data_loaded, haze_data_loaded, pretrained_path, pretrained=True, do_visualize=False)
