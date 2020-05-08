import cv2
import numpy as np
import torch
from torch.nn import functional as F


def preprocess_image(img, input_h, input_w):
    h, w, c = img.shape
    # print("image shape:", img.shape)
    # print("input wh:", input_w, input_h)
    h_r = input_h / h
    w_r = input_w / w
    if h_r > w_r:
        resized_other_length = int(h * w_r)
        short_length = input_h - resized_other_length
        pad_length = short_length // 2
        resize_wh = (input_w, resized_other_length)

        if short_length % 2 == 1:
            padding = [[pad_length, pad_length+1], [0, 0], [0, 0]]
        else:
            padding = [[pad_length, pad_length], [0, 0], [0, 0]]
    else:
        resized_other_length = int(w * h_r)
        short_length = input_w - resized_other_length
        pad_length = short_length // 2
        resize_wh = (resized_other_length, input_h)
        if short_length % 2 == 1:
            padding = [[0, 0], [pad_length, pad_length+1], [0, 0]]
        else:
            padding = [[0, 0], [pad_length, pad_length], [0, 0]]
    # print(padding)
    img = cv2.resize(img, (resize_wh))
    img = np.pad(img, padding, mode="constant", constant_values=0)

    return img


def preprocess_matrix(img, input_h, input_w):
    h, w = img.shape
    # print("image shape:", img.shape)
    # print("input wh:", input_w, input_h)
    h_r = input_h / h
    w_r = input_w / w
    if h_r > w_r:
        resized_other_length = int(h * w_r)
        short_length = input_h - resized_other_length
        pad_length = short_length // 2
        resize_wh = (input_w, resized_other_length)

        if short_length % 2 == 1:
            padding = [[pad_length, pad_length+1], [0, 0]]
        else:
            padding = [[pad_length, pad_length], [0, 0]]
    else:
        resized_other_length = int(w * h_r)
        short_length = input_w - resized_other_length
        pad_length = short_length // 2
        resize_wh = (resized_other_length, input_h)
        if short_length % 2 == 1:
            padding = [[0, 0], [pad_length, pad_length+1]]
        else:
            padding = [[0, 0], [pad_length, pad_length]]
    # print(padding)
    img = torch.FloatTensor(img[None, None, :, :])
    img = F.interpolate(img, size=resize_wh, mode="bilinear")
    img = np.pad(img[0, 0], padding, mode="constant", constant_values=0)

    return img
    