import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import RealDataSet, data_load_from_csv, nearest_idx, rse_idx, fle_idx, obtain_trigger_idx
from inferencing import Inferencing, sample_to_segments, segments_to_sample

if __name__ == '__main__':
    # file path
    data_path = './experimental_data/01272023/SA3_221_FcPSB_2um_E_compilednoname.csv'
    data_assembled = data_load_from_csv(data_path)
    # sample id in a file
    segment_idx = 0
    # potential mask for signal isolation
    pmask = 0.35
    # truncate data
    t_vec = data_assembled[:, segment_idx*3]/1e6
    p_vec = data_assembled[:, segment_idx*3+1]
    i_vec = data_assembled[:, segment_idx*3+2]
    (start_idx, end_idx) = obtain_trigger_idx(p_vec, pmask)
    t_vec = t_vec[start_idx : end_idx].copy()
    p_vec = p_vec[start_idx : end_idx].copy()
    i_vec = i_vec[start_idx : end_idx].copy()
    t_vec -= t_vec[0]
    # sample to segments
    x_data, y_data, leftover = sample_to_segments(i_vec)
    real_dataset = RealDataSet(x_data, y_data)
    data_loader = DataLoader(real_dataset, batch_size=1, num_workers=1, shuffle=False)
    infer_obj = Inferencing(device='cpu', model_stat_path='./model/training_model/fine_tunning_01062023_7.stat', thresh=0.5)
    infer_obj.set_data_loader(data_loader)
    x_result, y_result = infer_obj.infer_fn()
    y_recompose = segments_to_sample(y_result)
    t_vec_crop = t_vec[leftover:]
    i_vec_crop = i_vec[leftover:]
    i_max = np.amax(i_vec_crop)
    # figures
    offset_start = int(3.6/(5e-4))
    offset_end = int(4.3/(5e-4))
    plt.rcParams['figure.figsize'] = (10,3)
    fig1 = plt.figure(1)
    #plt.plot(t_vec_crop, i_vec_crop)
    #plt.plot(t_vec_crop, y_recompose*0.02+i_max)
    plt.plot(t_vec_crop[offset_start : offset_end], i_vec_crop[offset_start : offset_end])
    plt.plot(t_vec_crop[offset_start : offset_end], y_recompose[offset_start : offset_end]*0.02+i_max)
    plt.tight_layout()
    fig2 = plt.figure(2)
    ax = plt.axes()
    ax.set_axis_off()
    plt.plot(t_vec_crop[offset_start : offset_end], i_vec_crop[offset_start : offset_end], color='b')
    plt.savefig('./current_transparent.svg', format='svg', transparent=True)
    fig3 = plt.figure(3)
    ax = plt.axes()
    ax.set_axis_off()
    plt.plot(t_vec_crop[offset_start : offset_end], y_recompose[offset_start : offset_end], color='r')
    plt.savefig('./label_transparent.svg', format='svg', transparent=True)
    plt.show()