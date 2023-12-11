import os, sys, math, random, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from signal_generator import Collision_Combined_Signal

def data_load_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.values

def nearest_idx(data_arr, value):
    diff_arr = np.absolute(data_arr - value)
    nearest_idx = diff_arr.argmin()
    return (nearest_idx, data_arr[nearest_idx])

def rse_idx(bools):
    bools_rl = np.delete(np.insert(bools, obj=0, values=bools[0]), len(bools))
    signal_pulse = bools & (~bools_rl)
    return np.where(signal_pulse)[0]

def fle_idx(bools):
    bools_ll = np.delete(np.append(bools, bools[-1]), 0)
    signal_pulse = bools & (~bools_ll)
    return np.where(signal_pulse)[0]

def obtain_trigger_idx(trigger_arr, pmask):
    trigger_bools = trigger_arr > pmask
    start_idx = rse_idx(trigger_bools)[0]+1
    end_idx = fle_idx(trigger_bools)[0]
    return (start_idx, end_idx)

class DataComposer:
    def __init__(self, file_name='./data/training_dataset/default', sample_num=4):
        self.target_file_name = file_name
        self.target_sample_num = sample_num
        self.x_data = []
        self.y_data = []
        self.signal_data = np.array([])
        self.signal_label = np.array([])
    def compose(self):
        leftover = 0
        for iter in range(self.target_sample_num):
            signal_obj = Collision_Combined_Signal(fs=int(2e3), t=10)
            self.signal_data = signal_obj.signal
            self.signal_label = signal_obj.event_obj.event_label
            signal_len = len(signal_obj.signal)
            # running sliding window with 512 feeding size and 320 step size in backward manner
            for jter in range(signal_len-1, 512, -320):
                # squeeze signal to (0, 1)
                signal = signal_obj.signal[jter-511:jter+1].copy()
                signal_max = np.amax(signal)
                signal_min = np.amin(signal)
                signal = (signal-signal_min) / (signal_max-signal_min)
                self.x_data.append(np.expand_dims(signal, 0))
                self.y_data.append(np.expand_dims(signal_obj.event_obj.event_label[jter-511:jter+1], 0))
                leftover = jter-512
        return leftover
    def save(self):
        with open(self.target_file_name + '_x.pickle', 'wb') as pickle_out:
            pickle.dump(self.x_data, pickle_out)
            pickle_out.close()
        with open(self.target_file_name + '_y.pickle', 'wb') as pickle_out:
            pickle.dump(self.y_data, pickle_out)
            pickle_out.close()

class RealDataComposer:
    def __init__(self, src_name_list, tar_name='./data/training_dataset/real_default'):
        self.src_name_list = src_name_list
        self.tar_name = tar_name
        self.x_data = []
        self.y_data = []
        self.leftovers = []
    def compose(self):
        leftover = 0
        for iter in range(len(self.src_name_list)):
            # process signal
            data_assembled = data_load_from_csv(self.src_name_list[iter])
            (rows, cols) = data_assembled.shape
            for jter in range(cols//3):
                try:
                    p_vec = data_assembled[:, jter*3+1]
                    i_vec = data_assembled[:, jter*3+2]
                    (start_idx, end_idx) = obtain_trigger_idx(p_vec, 0.35)
                    i_vec = i_vec[start_idx : end_idx].copy()
                    signal_len = len(i_vec)
                    # running sliding window with 512 feeding size and 320 step size in backward manner
                    for kter in range(signal_len-1, 512, -320):
                        # squeeze signal to (0, 1)
                        signal = i_vec[kter-511:kter+1].copy()
                        signal_max = np.amax(signal)
                        signal_min = np.amin(signal)
                        signal = (signal-signal_min) / (signal_max-signal_min)
                        self.x_data.append(np.expand_dims(signal, 0))
                        self.y_data.append(np.expand_dims(np.zeros((512)), 0))
                        leftover = kter-512
                    self.leftovers.append(leftover)
                except:
                    pass
    def save(self):
        with open(self.tar_name + '_x.pickle', 'wb') as pickle_out:
            pickle.dump(self.x_data, pickle_out)
            pickle_out.close()
        with open(self.tar_name + '_y.pickle', 'wb') as pickle_out:
            pickle.dump(self.y_data, pickle_out)
            pickle_out.close()

class SampleDataSet(Dataset):
    def __init__(self, data_path='./data/training_dataset/default'):
        self.data_path = data_path
        with open(self.data_path+'_x.pickle', 'rb') as x_fp:
            self.x_data = pickle.load(x_fp)
            x_fp.close()
        with open(self.data_path+'_y.pickle', 'rb') as y_fp:
            self.y_data = pickle.load(y_fp)
            y_fp.close()
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

class RealDataSet(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

if __name__ == '__main__':
    data_composer = DataComposer(sample_num=1)
    data_composer.compose()
    data_composer.save()
    # show signal
    with open('./data/training_dataset/default_x.pickle', 'rb') as x_fp:
        x_data_loaded = pickle.load(x_fp)
        x_fp.close()
    with open('./data/training_dataset/default_y.pickle', 'rb') as y_fp:
        y_data_loaded = pickle.load(y_fp)
        y_fp.close()
    plt.figure(1)
    plt.plot(x_data_loaded[4])
    plt.show()