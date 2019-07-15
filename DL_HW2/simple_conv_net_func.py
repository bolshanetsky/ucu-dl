from __future__ import print_function
import torch
import numpy as np
from scipy import signal

def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    #input 64x1x28x28
    stride = 1
    batch_size, in_channels, heigth, width = x_in.shape
    out_channels = len(conv_weight)
    kernel_size = conv_weight.shape[2]
    s_out = int((heigth - kernel_size)/stride) + 1
    output = np.zeros((batch_size, out_channels, s_out, s_out), dtype=np.float32)

    #iterate over batch
    for n_batch in range(batch_size):
        image = x_in[n_batch][0]
        # iterate over filters
        for channel in range(out_channels):
            #iterate over 2D input
            for j in range(s_out):
                for i in range(s_out):
                    value = (torch.sum(image[i:i+kernel_size, j:j+kernel_size]*conv_weight[channel]) + conv_bias[channel]).item()
                    output[n_batch][channel][i][j] = value

    return torch.tensor(output, dtype=torch.float32)

def conv2d_vector(x_in, conv_weight, conv_bias, device):
    stride = 1
    batch_size, channels, heigth, width = x_in.shape
    out_channels = len(conv_weight)
    kernel_size = conv_weight.shape[2]
    s_out = int((heigth - kernel_size)/stride) + 1
    output = np.zeros((batch_size, out_channels, s_out, s_out), dtype=np.float32)
    for i, image in enumerate(x_in):
        A_col = im2col(image, kernel_size, stride, device)
        W_row = conv_weight2rows(conv_weight)
        Z = np.matmul(W_row, A_col) + conv_bias.detach().numpy()[:,np.newaxis]
        Z.resize((1,20, 24, 24))
        output[i] = Z
    
    return torch.tensor(output, dtype=torch.float32)

def im2col_x(X, kernel_size, stride, device):
    channels, height, width = X.shape
    s_out = int((height - kernel_size)/stride) + 1
    output = np.full((channels*s_out*s_out, kernel_size*kernel_size), np.nan, dtype=np.float32)
    for i in range(s_out):
        for j in range(s_out): 
            out_idx = j + i  * s_out
            value = X[:, i:i+kernel_size, j:j+kernel_size].numpy().flatten()
            output[out_idx] = value

    return output.transpose()

def im2col(X, kernel_size, stride, device):
    channels, height, width = X.shape
    s_out = int((height - kernel_size)/stride) + 1
    output = np.full((channels*s_out*s_out, kernel_size*kernel_size), np.nan, dtype=np.float32)
    for ch in range(channels):
        for i in range(s_out):
            for j in range(s_out): 
                out_idx = j + i  * s_out + ch*(s_out*s_out)
                value = X[ch, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size].detach().numpy().flatten()
                output[out_idx] = value

    return output.transpose()    


def conv_weight2rows(conv_weight):
    output = np.zeros((20, 25))
    for i in range(len(conv_weight)):
        output[i] = conv_weight[i, 0].detach().numpy().flatten()

    return output


def pool2d_scalar(a, device):
    batch_size = a.shape[0]
    output = np.zeros((batch_size, 20, 12, 12))
    for n_batch in range(batch_size):
        for c_out in range(20):
            for i in range(12):
                for j in range(12):
                    output[n_batch][c_out][i][j] = torch.max(a[n_batch, c_out, i*2:i*2+2, j*2:j*2+2])
    
    return torch.tensor(output, dtype=torch.float32)


def pool2d_vector(a, device):
    batch_size, c_in, heigth, width = a.shape
    stride = 2
    kernel_size = 2
    s_out = int(c_in*heigth*width/(kernel_size*kernel_size))
    output = np.full((batch_size, s_out), np.nan, dtype=np.float32)
    for i in range(batch_size):
        test = im2col(a[i], kernel_size, stride, device)
        output[i] = np.amax(im2col(a[i], kernel_size, stride, device), axis=0)
    
    return torch.tensor(output, dtype=torch.float32)


def relu_scalar(a, device):
    for i in range(a.shape[1]):
        a[0][i] = max(0, a[0][i].item())
    return a


def relu_vector(a, device):
    return torch.clamp(a, min = 0)


def reshape_vector(a, device):
    ## Skipped, reshape done on pooling layer.
    pass

def fc_layer_scalar(a, weight, bias, device):
    batch_size, input_length = a.shape
    output_length = weight.shape[0]
    output = np.zeros((batch_size, output_length))

    for n in range(batch_size):
        for j in range(output_length):
            sum = 0
            for i in range(input_length):
                sum+= weight[j, i] * a[n, i]
            output[n, j] = sum + bias[j]
    
    return torch.tensor(output, dtype=torch.float32)


def fc_layer_vector(a, weight, bias, device):
    batch_size = a.shape[0]
    output_length = weight.shape[0]
    output = np.full((batch_size, output_length), np.nan, dtype=np.float32)
    output = torch.from_numpy(output)
    for n in range(batch_size):
        output[n] = torch.matmul(weight, a[n]) + bias
    return output