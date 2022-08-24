# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:05:01 2021

@author: bernhard
"""

import numpy as np

#%% Dimensional transformations

def conv_2d_4d(A):
    '''
    conv_2d_4d() converts a 4D transformation matrix with the spaitial dimensions (x_out, y_out, x_in, y_in) into a 2D where input and output are reshaped into single dimensions (out,in).
    It works the same way in the other direction, from 2D to 4D.

    Parameters
    ----------
    A : Matrix with 4 or 2 dimensions.

    Returns
    -------
    A_out : Reshaped matrix with 2 or 4 dimensions, respectively.

    '''
    data_type = A.dtype
    
    # 4D to 2D transformation
    if A.ndim == 4:
        
        A_temp = np.zeros((A.shape[0]*A.shape[1],A.shape[2],A.shape[3]),dtype=data_type)
        for i in range(A.shape[2]):
            for j in range(A.shape[3]):
                A_temp[:,i,j] = np.reshape(A[:,:,i,j], (A.shape[0]*A.shape[1],))
                
        A_out = np.zeros((A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]),dtype=data_type)
        for i in range(A_out.shape[0]):
            A_out[i,:] = np.reshape(A_temp[i,:,:], (A.shape[2]*A.shape[3],))
    
    # 2D to 4D transformation
    elif A.ndim == 2:
        
        A_temp = np.zeros((np.sqrt(A.shape[0]),np.sqrt(A.shape[0]),A.shape[1]),dtype=data_type)
        for i in range(A.shape[1]):
            A_temp[:,:,i] = np.reshape(A[:,i], (np.sqrt(A.shape[0]),np.sqrt(A.shape[0])))
            
        A_out = np.zeros((np.sqrt(A.shape[0]),np.sqrt(A.shape[0]),np.sqrt(A.shape[1]),np.sqrt(A.shape[1])),dtype=data_type)
        for i in range(A_out.shape[2]):
            for j in range(A_out.shape[3]):
                A_out[i,j,:,:] = np.reshape(A_temp[i,j,:], (np.sqrt(A.shape[0]),np.sqrt(A.shape[0])))
                
    else:
        
        A_out = np.copy(A)
            
    return A_out

#%% Transformation matrices (2D output - 2D input)

def fft_mat(N, zero_pad_fact):
    '''
    fft_mat() returns a reshaped 2D matrix that implements the 2D FFT (fft2) of an image that was reshaped into a single dimension vector.
    Variable zero padding can be applied to customize the number of output pixels.
    The zeros at the input dimension are cut from the matrix.
    The typically necessary fftshift, bringing the low frenquencies to the center, is taken care of.

    Parameters
    ----------
    N : number of input modes along one spatial axis, i.e. N=16 results in 16x16 = 256 input modes arranged in a square.
    zero_pad_fact : amount of zero padding. zero_pad_fact = 0 means no zero padding, the output dimensions are the same as the input dimensions. zero_pad_fact = 1 means a quadrupling of the output dimesions, i.e. for N=16 it leads to N_out = 32 so 1024 output modes for 256 input modes. Any value inbetween 0 and 1 or higher is valid as well.

    Returns
    -------
    output : 2D fft2 matrix

    '''
    
    # get padded dimension length and indices of the central unpadded part
    N_pad = int(N + 2*round(N*zero_pad_fact/2))
    x_pad = np.arange(int(round(N*zero_pad_fact/2)),int(N+round(N*zero_pad_fact/2)))

    # span 4D meshgrid
    j = np.arange(-N_pad/2,N_pad/2)
    J, P, K, Q = np.meshgrid(j,j,j,j)
    
    # 4D matrix representing fftshift(fft2())
    exp_mat = np.exp(-1j * (2*np.pi/N_pad) * J*P) * np.exp(-1j * (2*np.pi/N_pad) * K*Q)
    exp_mat = np.moveaxis(exp_mat, [0, 1 ,2, 3], [1, 3, 2, 0])    

    # selection of the upadded central input region
    exp_mat_cut = exp_mat[:,:,x_pad,:][:,:,:,x_pad]
    
    # reshaping to a 2D array before return
    return conv_2d_4d( exp_mat_cut )


def grain_mat(N,sigma):
    '''
    grain_mat() returns a 2D matrix that implements a convolution with a Gaussian smearing kernal for a 2D image that is reshaped to a single dimension vector.

    Parameters
    ----------
    N : number of modes/elements along one spatial dimension. For N=16 a 256x256 matrix is returned that smears a 16x16 image.
    sigma : width of the Gaussian smearing kernal in pixel

    Returns
    -------
    output : 2D matrix implementing the 2D convolution with a Gaussian

    '''
    
    # span 4D meshgrid
    j = np.arange(-N/2,N/2)
    J, P, K, Q = np.meshgrid(j,j,j,j)
    
    # 4D matrix implementing the 2D smearing of an image
    exp_mat = np.exp(-(J-P)**2/(2*sigma**2)) * np.exp(-(K-Q)**2/(2*sigma**2))
    exp_mat = np.moveaxis(exp_mat, [0, 1 ,2, 3], [1, 3, 2, 0])    

    # reshaping to a 2D array before return
    return conv_2d_4d( exp_mat )

def gauss_diag(N,sigma):
    '''
    gauss_diag() returns a Gaussian diagonal matrix for a 2D input and a 2D output, reshaped into a 2D matrix. 

    Parameters
    ----------
    N : number of modes/elements along one spatial dimension, i.e. N=16 returns a 256x256 matrix.
    sigma : width of the Gaussian diagnonal in pixel

    Returns
    -------
    output : 2D matrix representing the reshaped version of 4D Gaussian diagonal matrix

    '''
    
    if sigma > 0:
        
        # smear kernal 
        smear_kernal = grain_mat(N,sigma)
        
        # fill each output row with a smeared and reshaped version of single input pixel
        output = np.zeros((N**2,N**2))
        for i in range(N**2):
            
            mask_in = np.zeros((N**2,))
            mask_in[i] = 1
            
            mask_out = smear_kernal @ mask_in
            output[:,i] = mask_out / np.sum(mask_out)
    
    else:
        
        # for sigma = 0 it returns simply a diagonal matrix
        output = np.diag(np.ones((N**2,)))
            
    return output

#%% Transformation matrices (1D output - 1D input)

def fft_mat_1d(N):
    '''
    fft_mat_1d() retruns a matrix which implements the Fourier transform and shift of a 1D vector.

    Parameters
    ----------
    N : number of input modes

    Returns
    -------
    output : 1D fftshift(fft()) matrix

    '''
    
    # span 2D meshgrid
    j = np.arange(-N/2,N/2)
    J, P = np.meshgrid(j,j)
    
    # 2D matrix representing fftshift(fft())
    output = np.exp(-1j * (2*np.pi/N) * J*P)
            
    return output

def grain_mat_1d(N,sigma):
    '''
    grain_mat_1d() returns a 2D matrix that implements a convolution with a Gaussian smearing kernal for a 1D vector.

    Parameters
    ----------
    N : number of modes/elements along the spatial dimension.
    sigma : width of the Gaussian smearing kernal in pixel

    Returns
    -------
    output : 2D matrix implementing the 1D convolution with a Gaussian

    '''
    
    # span 2D meshgrid
    j = np.arange(-N/2,N/2)
    J, P = np.meshgrid(j,j)
    
    # 2D matrix implementing the 1D smearing of a vector
    output = np.exp(-(J-P)**2/(2*sigma**2))
            
    return output

#%% Transmission matrices

def trans_me_custom(r, zero_pad_fact):
    '''
    trans_me_custom() returns a infinite memory effect TM contructed from defined phase and amplitude mask r

    Parameters
    ----------
    r : custom phase and amplitude mask reshaped into a 1D vector
    zero_pad_fact : amount of zero padding (see fft_mat())

    Returns
    -------
    T_me : infinite memory effect TM

    '''
    
    # get input size
    N = int(np.sqrt(r.size))    

    # contruct TM from fft of diagonal matrix
    T_me = fft_mat(N,zero_pad_fact)  @ np.diag(r)
    
    return T_me

def trans_me(N, zero_pad_fact, sigma_me, sigma_grain):
    '''
    trans_me() 

    Parameters
    ----------
    N : number of input modes along one spatial axis, i.e. N=16 results in 16x16 = 256 input modes arranged in a square.
    zero_pad_fact : amount of zero padding (see fft_mat())
    sigma_me : width of the memory effect decorrelation in pixel. sigma_me = 0 corresponds to infinite memory effect while sigma_me > N/2 corresponds to no memory effect.
    sigma_grain : width of additional output grain smearing in pixel

    Returns
    -------
    T_me : memory effect TM

    '''
    
    # random Gaussian diagonal matrix 
    rand_gauss_diag = gauss_diag(N,sigma_me) * (np.random.normal(size=(N**2,N**2)) + 1j * np.random.normal(size=(N**2,N**2)))
    
    # contruct TM from fft of diagonal matrix
    T_me = fft_mat(N, zero_pad_fact) @ rand_gauss_diag
    
    # apply additional smearing at the output
    if sigma_grain > 0:
        T_me = grain_mat(int(np.sqrt(T_me.shape[0])),sigma_grain) @ T_me
    
    return T_me
