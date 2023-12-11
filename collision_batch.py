import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import RealDataSet, data_load_from_csv, nearest_idx, rse_idx, fle_idx, obtain_trigger_idx
from inferencing import Inferencing, sample_to_segments, segments_to_sample

def piecewise_search(y):
    # rse and fle indexing
    y_bool_arr = np.bool_(y)
    y_rse_idx = rse_idx(y_bool_arr)
    y_fle_idx = fle_idx(y_bool_arr)
    return (y_rse_idx, y_fle_idx)

def crop_data_series(data_assembled, segment_idx):
    t_vec = data_assembled[:, segment_idx*3]/1e6
    p_vec = data_assembled[:, segment_idx*3+1]
    i_vec = data_assembled[:, segment_idx*3+2]
    (start_idx, end_idx) = obtain_trigger_idx(p_vec, 0.35)
    t_vec = t_vec[start_idx : end_idx].copy()
    p_vec = p_vec[start_idx : end_idx].copy()
    i_vec = i_vec[start_idx : end_idx].copy()
    t_vec -= t_vec[0]
    return (t_vec, p_vec, i_vec)

def plateau_baseline_avg(i_vec, rse, fle):
    # baseline avg containers
    avg_head_buf = np.zeros((len(y_rse_idx)))
    avg_tail_buf = np.zeros((len(y_fle_idx)))
    i_bound = len(i_vec) - 1
    avg_bound = 8
    offset = 4
    for iter in range(len(y_rse_idx)):
        head_avg_idx = rse[iter] - avg_bound - offset
        head_avg_idx = head_avg_idx if head_avg_idx >= 0 else 0
        tail_avg_idx = fle[iter] + avg_bound + offset
        tail_avg_idx = tail_avg_idx if tail_avg_idx <= i_bound else i_bound
        avg_head_buf[iter] = np.mean(i_vec[head_avg_idx:rse[iter]-offset])
        avg_tail_buf[iter] = np.mean(i_vec[fle[iter]+offset:tail_avg_idx])
    return (avg_head_buf, avg_tail_buf)

def i_step_function(i_vec, event_height_vec, rse, fle):
    i_pulse = np.zeros((len(i_vec)))
    for iter in range(len(rse)):
        i_pulse[(rse[iter]+fle[iter])//2] = event_height_vec[iter]
    i_step = np.add.accumulate(i_pulse)
    return i_step

if __name__ == '__main__':
    data_path = './experimental_data/01272023/SA3_221_FcPSB_2um_E_compilednoname.csv'
    tar_path = './experimental_data/01272023/output'
    tar_wave_fp = tar_path + '_waveform.csv'
    tar_peak_fp = tar_path + '_peaks.csv'
    data_assembled = data_load_from_csv(data_path)
    (rows, cols) = data_assembled.shape
    sample_num = cols//3
    #sample_num = 120
    sample_start = 0
    # current threshold
    i_suppressor = 1.5e-3
    fs = (data_assembled[1, 0] - data_assembled[0, 0])/1e6
    with open(tar_wave_fp, 'w', newline='') as hwave, open(tar_peak_fp, 'w', newline='') as hpeak:
        wave_writer = csv.writer(hwave, delimiter=',')
        peak_writer = csv.writer(hpeak, delimiter=',')
        for iter in range(sample_start, sample_num, 1):
            print(f'Processing electrode {iter+1-sample_start}/{sample_num-sample_start}:')
            # extract and crop data series
            (t_vec, p_vec, i_vec) = crop_data_series(data_assembled, iter)
            # sample to segments
            x_data, y_data, leftover = sample_to_segments(i_vec)
            # dataset and loaders
            dataset = RealDataSet(x_data, y_data)
            data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
            # inferencing
            infer_obj = Inferencing(device='cpu', model_stat_path='./model/training_model/fine_tunning_01062023_7.stat', thresh=0.5)
            infer_obj.set_data_loader(data_loader)
            x_result, y_result = infer_obj.infer_fn()
            # segments back to sample array
            y_recompose = segments_to_sample(y_result)
            # rescale vectors and crop t and i vectors
            t_vec = t_vec[leftover:]
            i_vec = i_vec[leftover:]
            # piecewise event search
            (y_rse_idx, y_fle_idx) = piecewise_search(y_recompose)
            # check and coerce the length of rse and fle
            rse_len = len(y_rse_idx)
            fle_len = len(y_fle_idx)
            if rse_len > fle_len:
                y_rse_idx = np.delete(y_rse_idx, np.s_[0:(rse_len-fle_len)])
            if fle_len > rse_len:
                y_fle_idx = np.delete(y_fle_idx, np.s_[0:(fle_len-rse_len)])
            if len(y_rse_idx) > 0:
                # baseline avg
                (avg_head, avg_tail) = plateau_baseline_avg(i_vec, y_rse_idx, y_fle_idx)
                event_height_vec = avg_tail - avg_head
                # suppress insignificant events
                suppressed_idx = np.where(event_height_vec < i_suppressor)[0]
                event_height_vec = np.delete(event_height_vec, suppressed_idx)
                y_rse_idx = np.delete(y_rse_idx, suppressed_idx)
                y_fle_idx = np.delete(y_fle_idx, suppressed_idx)
                avg_head = np.delete(avg_head, suppressed_idx)
                # i step function
                i_step = i_step_function(i_vec, event_height_vec, y_rse_idx, y_fle_idx)
                # t event
                t_event = (y_rse_idx + y_fle_idx) / 2.0 * fs + t_vec[0]
                # current drop ratio
                i_drop_ratio = event_height_vec / avg_head
                # save files
                wave_writer.writerows(np.stack((t_vec, i_vec, i_step)))
                peak_writer.writerows(np.stack((t_event, event_height_vec, avg_head, i_drop_ratio)))
            else:
                wave_writer.writerows(np.stack(([],[],[])))
                peak_writer.writerows(np.stack(([],[],[],[])))
            hwave.flush()
            hpeak.flush()
        hwave.close()
        hpeak.close()
    print('Done.')

