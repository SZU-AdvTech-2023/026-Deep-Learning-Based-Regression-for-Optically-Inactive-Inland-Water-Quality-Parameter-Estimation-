import ConvNext
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import imageio.v3 as imageio
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import joblib
import matplotlib.colors as mcolors


parser = argparse.ArgumentParser(description='Convnext_pic_args')
parser.add_argument('--model_path', type=str, default='./model/', help='model_path')
parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu_id')
parser.add_argument('--hyp_path', type=str, default='')
parser.add_argument('--element_id', type=int, default=10, help='TP:10,TN:11,NH3:12')
args = parser.parse_args()


def get_element_name(element_id):  # 获取元素名的函数
    if element_id == 10:
        element_name = 'TP'
    elif element_id == 11:
        element_name = 'TN'
    elif element_id == 12:
        element_name = 'NH3'
    return element_name


def water_part(hyp_path, green=49, nir=158, ndwi_thresh=0.3):  # 这个函数通过NDWI获取是水体的所有点的(x,y)坐标
    data = []
    label = []
    river = imageio.imread(hyp_path)
    # ndwi 水体提取
    for i in tqdm(range(0, 2789)):  # tqdm会显示出循环运行的进度
        for j in range(0, 3718):
            # np.any()对矩阵所有元素做或运算，目的是判断矩阵是否存在不为0的元素，存在True则返回True;
            # river[:,i,j]是切片操作，返回第二个维度的第i个元素的第j个元素
            if np.any(river[:, i, j]):
                # append()是将(i,j)插入现有列表中
                label.append((i, j))
                # 同理，data.append是把上面提取到的元素放到data列表中
                data.append(river[:, i, j])

    data_arr = np.asarray(data)  # 将data转换为数组
    # 取数组中所有绿色波段减去近红外波段除以绿色波段加近红外波段。（ndwi公式）
    ndwi = (data_arr[:, green] - data_arr[:, nir]) / (data_arr[:, green] + data_arr[:, nir])

    # 以阈值为 ndwi_thresh 进行提取
    water_index = []
    water_data = []
    for i in tqdm(range(0, len(label))):
        if ndwi[i] > ndwi_thresh:
            water_index.append(label[i])
            water_data.append(data[i])
    print(len(water_index))

    final_index = water_index
    np_final_river = np.asarray(water_data)
    river_data = np_final_river

    # 绘制水体图像
    watar_map = np.zeros((2839, 3718))  # 绘制一个1626*1524的全0矩阵
    for index in final_index:
        watar_map[index] = 1  # final_index（水体），置为1

    plt.matshow(watar_map, cmap=plt.cm.Blues)  # 填充蓝色
    plt.savefig(fr'./水体提取图.png', dpi=600)
    plt.show()
    return final_index


def extract_image_block(x, y, image_path=args.hyp_path):
    with rasterio.open(image_path) as src:
        # 创建一个窗口，以(x, y)为中心，大小为32*32
        window = Window(x - 16, y - 16, 32, 32)

        # 读取图像数据
        img_data = src.read(window=window)

    return img_data


def extract_pixel_values(x, y, band_numbers, hyp_img):
    # 将波段号列表中的值减 1
    band_numbers = [band - 1 for band in band_numbers]
    # 提取特定坐标点对应的波段信息
    pixel_values = hyp_img[y, x, band_numbers]
    pixel_values = np.array(pixel_values).reshape(1, -1)
    return pixel_values


if __name__ == '__main__':
    hyp_img = imageio.imread(args.hyp_path)  # 读取遥感图像
    pearson_tp = [133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                  153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
                  173, 175, 176, 177, 178, 179, 180, 181, 185, 186, 187, 188, 189, 199, 200, 201, 202]
    extract_pixel_values(79, 54, pearson_tp, hyp_img)
    water = water_part(args.hyp_path)  # 获得水体部分的坐标
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model = ConvNext.convnext_tiny(5)
    model.to(device)
    state_dict = torch.load(f'{args.model_path}{get_element_name(args.element_id)}/ConvNext_best.pth')
    model.load_state_dict(state_dict['model'])
