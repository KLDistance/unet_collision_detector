import os, sys, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from tqdm import tqdm
from unet_model import UNET
from dataset import DataComposer, SampleDataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Inferencing:
    def __init__(self, device='cpu', model_stat_path='./model/training_model/fine_tunning_12022022_1.stat', thresh=0.5):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.thresh = thresh
        self.model_stat_path = model_stat_path
        self.model = UNET()
        self.model.load_state_dict(torch.load(self.model_stat_path, map_location=torch.device(self.device))['state_dict'])
    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
    def infer_fn(self):
        x_list = []
        y_list = []
        loop = tqdm(self.data_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, y_dummy) in enumerate(loop):
                x = x.float().to(device=self.device)
                preds = torch.sigmoid(self.model(x))
                preds = (preds > self.thresh).float()
                x_list.append(x.detach().cpu().numpy()[0, 0, :])
                y_list.append(preds.detach().cpu().numpy()[0, 0, :])
        return x_list, y_list

def sample_to_segments(data):
    signal_len = len(data)
    leftover = 0
    x_data = []
    y_data = []
    # running sliding window with 512 feeding size and 320 step size in backward manner
    for jter in range(signal_len-1, 512, -320):
        # squeeze signal to (0, 1)
        signal = data[jter-511:jter+1].copy()
        signal_max = np.amax(signal)
        signal_min = np.amin(signal)
        signal = (signal-signal_min) / (signal_max-signal_min)
        x_data.append(np.expand_dims(signal, 0))
        y_data.append(np.expand_dims(np.zeros((signal_len)), 0))
        leftover = jter-512
    return x_data, y_data, leftover

def segments_to_sample(segments):
    segment_num = len(segments)
    pixel_num = segment_num * 320 + 192 + 1
    signal = np.zeros((pixel_num))
    for iter in range(segment_num):
        signal[pixel_num-512-320*iter:pixel_num-320*iter] = segments[iter]
    return signal

def virtual_data_inferencing():
    infer_obj = Inferencing(device='cpu', model_stat_path='./model/training_model/fine_tunning_01062023_1.stat')
    # generate data
    data_composer = DataComposer(file_name='./data/testing_dataset/test1', sample_num=1)
    leftover = data_composer.compose()
    data_composer.save()
    # generate dataset
    test_dataset = SampleDataSet(data_path='./data/testing_dataset/test1')
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    # inference
    infer_obj.set_data_loader(test_data_loader)
    x_list, y_list = infer_obj.infer_fn()
    original_vec = data_composer.signal_data[leftover:-1]
    #original_vec = segments_to_sample(x_list)
    preds_vec = segments_to_sample(y_list)
    valid_vec = data_composer.signal_label[leftover:-1]
    plt.figure(1)
    plt.plot(original_vec)
    plt.plot(preds_vec*0.2)
    plt.plot(valid_vec*0.1)
    plt.show()

def real_data_inferencing():
    pass

if __name__ == '__main__':
    virtual_data_inferencing()