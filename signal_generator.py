import os, sys, math, random
import numpy as np
from scipy import optimize, signal, fft

class Collision_Combined_Signal:
    def __init__(self, fs, t):
        self.fs = fs
        self.t = t
        self.noise_obj = Collision_Noise(fs=fs, t=t, mean=0, std=20e-4)
        self.capa_obj = Collision_Capacitance(fs=fs, t=t, mag=1, shift=5)
        self.event_obj = Collision_Events(fs=fs, t=t)
        self.event_obj.append_slope_events(event_density=9e-4, event_intensity=1.4e-3, event_px_width_range=80)
        self.signal = self.noise_obj.noise_vec + self.capa_obj.capa_vec + self.event_obj.event_vec

class Collision_Events:
    def __init__(self, fs, t):
        self.fs = fs
        self.t = t
        self.sample_num = round(fs*t)
        self.t_vec = np.linspace(0, t, self.sample_num)
        self.event_vec = np.zeros((self.sample_num))
    def append_impulse_events(self, event_density=1e-2, event_intensity=1e-3):
        self.impulse_event_density = event_density
        event_impulse_noise_pattern = np.random.normal(0, event_intensity, self.sample_num)
        event_sorted_impulse_noise_pattern = np.sort(event_impulse_noise_pattern)[::-1]
        event_impulse_threshold = event_sorted_impulse_noise_pattern[round(event_density*self.sample_num)]
        event_impulses = event_impulse_noise_pattern.copy() - event_impulse_threshold
        event_impulses[event_impulses < event_intensity*(random.random()/5+0.02)] = 0.0
        self.event_vec += np.add.accumulate(event_impulses)
        # event label
        self.event_label = event_impulses.copy()
        self.event_label[self.event_impulse_label > 0] = 1
    def append_slope_events(self, event_density=1e-3, event_intensity=1e-3, event_px_width_range=5):
        self.slope_event_density = event_density
        event_impulse_noise_pattern = np.random.normal(0, event_intensity, self.sample_num)
        event_sorted_impulse_noise_pattern = np.sort(event_impulse_noise_pattern)[::-1]
        event_impulse_threshold = event_sorted_impulse_noise_pattern[round(event_density*self.sample_num)]
        event_impulses = event_impulse_noise_pattern.copy() - event_impulse_threshold
        event_impulses[event_impulses < event_intensity*(random.random()/5+0.02)] = 0.0
        slope_event_kernel = np.ones((random.randint(25, event_px_width_range)))
        slope_event_conv = signal.convolve(event_impulses, slope_event_kernel)[0:self.sample_num]
        self.event_vec += np.add.accumulate(slope_event_conv)
        # event label with edge points included
        self.event_label = slope_event_conv.copy()
        self.event_label_right = np.roll(self.event_label, 1)
        self.event_label_right[0] = 0
        self.event_label_left = np.roll(self.event_label, -1)
        self.event_label_left[-1] = 0
        self.evelt_label = self.event_label + self.event_label_right + self.event_label_left
        self.event_label[slope_event_conv > 0] = 1

class Collision_Capacitance:
    def __init__(self, fs, t, mag, shift):
        self.fs = fs
        self.t = t
        self.mag = mag
        self.shift = shift
        self.sample_num = round(fs*t)
        self.t_vec = np.linspace(0, t, self.sample_num)
        self.capa_vec = -8*np.exp(-self.t_vec*32*shift)-\
            4*np.exp(-self.t_vec*16*shift)-\
            1.5*np.exp(-self.t_vec*5*shift)-\
            1*np.exp(-self.t_vec*0.7*shift)
        capa_min = abs(np.amin(self.capa_vec))
        self.capa_vec /= (capa_min/mag)

class Collision_Noise:
    def __init__(self, fs, t, mean, std):
        self.fs = fs
        self.t = t
        self.mean = mean
        self.std = std
        self.sample_num = round(fs*t)
        # time vector (sec)
        self.t_vec = np.linspace(0, t, self.sample_num)
        # noise vector
        self.noise_vec = np.random.normal(mean, std, self.sample_num)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fs = int(2e3)
    t = 10 # 10 sec
    noise_obj = Collision_Noise(fs=fs, t=t, mean=0, std=20e-4)
    capa_obj = Collision_Capacitance(fs=fs, t=t, mag=1, shift=5)
    event_obj = Collision_Events(fs=fs, t=t)
    #event_obj.append_impulse_events(event_density=1e-3, event_intensity=5e-3)
    event_obj.append_slope_events(event_density=9e-4, event_intensity=1.4e-3, event_px_width_range=80)
    fig1 = plt.figure('gaussian noise')
    plt.plot(noise_obj.t_vec, noise_obj.noise_vec)
    fig2 = plt.figure('capacitance')
    plt.plot(capa_obj.t_vec, capa_obj.capa_vec)
    fig3 = plt.figure('events')
    plt.plot(event_obj.t_vec, event_obj.event_vec)
    fig4 = plt.figure('combined signal')
    plt.plot(event_obj.t_vec, noise_obj.noise_vec+capa_obj.capa_vec+event_obj.event_vec)
    plt.show()