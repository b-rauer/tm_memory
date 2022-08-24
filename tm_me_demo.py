#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:52:37 2022

@author: bernhard
"""
# Run this to plot in external window allowing coninous plotting in for loop:
# %matplotlib qt

import numpy as np
import matplotlib.pyplot as plt
import tm_me_fnk

# input
N = 16
zero_pad_fact = 2
sigma_me = 4
sigma_out = 0

# construct TM
T_me = tm_me_fnk.trans_me(N, zero_pad_fact, sigma_me, sigma_out)
# T_me = tm_me_fnk.trans_me_custom(np.random.normal(size=(N**2,)) + 1j * np.random.normal(size=(N**2,)), zero_pad_fact)

# get input vector to focus at the output center
N_out = int(np.sqrt(T_me.shape[0]))
foc_idx = int(N_out**2/2 + N_out/2)
foc_mask = np.zeros((N_out**2,)) 
foc_mask[foc_idx] = 1
foc_vec = T_me.conj().transpose() @ foc_mask

# spatial grid for phase ramp
phi_max = 2 # range of phase ramp
x = np.arange(-N/2,N/2) / N
X, Y = np.meshgrid(x,x)
phase_tilt = np.linspace(-phi_max,phi_max,30)

# initialize plot window
fig, ax = plt.subplots()
im_handle = ax.imshow( np.ones((N_out,N_out)) )

# loop through phase ramps
for i in range(phase_tilt.size):
    
    # define current phase ramp
    phase_grad = np.exp(1j * 2*np.pi * (X-Y) * phase_tilt[i])
    
    # define input vector (focusing or plane wave)
    vec_in = foc_vec * np.reshape(phase_grad, (N**2,))
    # vec_in = np.ones((N**2,), dtype=complex) * np.reshape(phase_grad, (N**2,))

    # vec_out = T_me @ vec_in
    vec_out = T_me @ vec_in

    # plot output intensity
    im_handle.set_data( np.abs( np.reshape(vec_out, (N_out,N_out)) )**2 )
    im_handle.set_clim(vmin=0, vmax=np.max(np.abs( np.reshape(vec_out, (N_out,N_out)) )**2))
    plt.pause(0.2)
    