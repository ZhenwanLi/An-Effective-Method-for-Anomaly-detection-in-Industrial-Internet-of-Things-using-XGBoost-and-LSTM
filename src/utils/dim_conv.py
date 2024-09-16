# -*- coding: UTF-8 -*-
"""
@author:zhenwan
@file_name:dim_conv.py
@time:2023/04/07 1:23 PM
@IDE:PyCharm
@copyright:网安
"""
from math import sqrt

import torch
from torch import nn


def zerodim2onedim(x, num_old_features, num_new_features):
    batch_size, input_data = x.size()
    # Add 8 zero-valued features to the input tensor .cuda()
    zero_features = torch.zeros(batch_size, num_new_features).to(x.device)
    new_data = torch.cat((x, zero_features), dim=1)
    # Reshape the input tensor to have dimensions [batch_size, channels, height, width]
    input_size = int(num_old_features + num_new_features)
    new_data = new_data.view(batch_size, 1, input_size)
    # Print the shape of the input tensor
    # print(new_data.shape)
    # print(type(new_data))
    return new_data


def zerodim2twodim(x, num_old_features, num_new_features):
    batch_size, input_data = x.size()
    # Add 8 zero-valued features to the input tensor .cuda()
    zero_features = torch.zeros(batch_size, num_new_features).to(x.device)
    new_data = torch.cat((x, zero_features), dim=1)
    # Reshape the input tensor to have dimensions [batch_size, channels, height, width]
    h = w = int(sqrt(num_old_features + num_new_features))
    new_data = new_data.view(batch_size, 1, h, w)
    # Print the shape of the input tensor
    # print(new_data.shape)
    # print(type(new_data))
    return new_data


def onedim2twodim(x, num_old_features, num_new_features):
    batch_size, input_dim, input_size = x.size()
    # Add 8 zero-valued features to the input tensor .cuda()
    zero_features = torch.zeros(batch_size, input_dim, num_new_features).to(x.device)
    new_data = torch.cat((x, zero_features), dim=2)
    # Reshape the input tensor to have dimensions [batch_size, channels, height, width]
    h = w = int(sqrt(num_old_features + num_new_features))
    new_data = new_data.view(batch_size, input_dim, h, w)
    # Print the shape of the input tensor
    # print(new_data.shape)
    # print(type(new_data))
    return new_data


def twodim2onedim(x):
    batch_size, input_dim, h, w = x.size()

    new_data = x.view(batch_size, input_dim, h * w)
    # Print the shape of the input tensor
    # print(new_data.shape)
    # print(type(new_data))
    return new_data


# Define MultiKernelConv1d layer with kernels of different sizes
class MultiKernelConv1d(nn.Module):
    """
    kernel_sizes: please use odd list
    """

    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MultiKernelConv1d, self).__init__()
        self.conv_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ) for kernel_size in kernel_sizes]
        )

    def forward(self, x):
        out = []
        for conv_layer in self.conv_layers:
            out.append(conv_layer(x))
        return torch.cat(out, dim=1)


# Define MultiKernelConv2d layer with kernels of different sizes
class MultiKernelConv2d(nn.Module):
    """
    kernel_sizes: please use odd list
    """

    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(MultiKernelConv2d, self).__init__()
        self.conv_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ) for kernel_size in kernel_sizes]
        )

    def forward(self, x):
        out = []
        for conv_layer in self.conv_layers:
            out.append(conv_layer(x))
        return torch.cat(out, dim=1)


class MultiKernelConv1dFlatten(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1, padding=0):
        super(MultiKernelConv1dFlatten, self).__init__()
        self.conv_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Flatten(),
            ) for kernel_size in kernel_sizes]
        )

    def forward(self, x):
        out = []
        for conv_layer in self.conv_layers:
            out.append(conv_layer(x))
        return torch.cat(out, dim=1)


class MultiKernelConv2dFlatten(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1, padding=0):
        super(MultiKernelConv2dFlatten, self).__init__()
        self.conv_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Flatten(),
            ) for kernel_size in kernel_sizes]
        )

    def forward(self, x):
        out = []
        for conv_layer in self.conv_layers:
            out.append(conv_layer(x))
        return torch.cat(out, dim=1)


if __name__ == '__main__':
    x = torch.randn(32, 1, 7, 7)
    conv1 = nn.Conv2d(1, 3, 9, padding=9 // 2)
    out = conv1(x)
    print(out.shape)

    print(5 // 2)

    x = torch.randn(32, 41)
    x = zerodim2twodim(x, 41, 8)
    print(x.shape)

    x = torch.randn(32, 47)
    x = zerodim2onedim(x, 47, 2)
    print(x.shape)

    x = torch.randn(32, 5, 41)
    x = onedim2twodim(x, num_old_features=41, num_new_features=8)
    print(x.shape)

    x = torch.randn(32, 1, 7, 7)
    x = twodim2onedim(x)
    print(x.shape)
