import DNNRNet_point
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


parser = argparse.ArgumentParser(description='DNNRNet_args')
parser.add_argument('--train_batch_size', type=int, default=30, help='train_batch_size')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate for optimizer')
parser.add_argument('--in_channels', type=int, default=216, help='Number of input channels')
parser.add_argument('--train_data', type=str, default='', help='Train data Loc')
parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu_id')
parser.add_argument('--element_id', type=int, default=10, help='TP:10,TN:11,NH3:12')
parser.add_argument('--excel_path', type=str, default='')
parser.add_argument('--save_model_path', type=str, default='./model', help='path to save model')
parser.add_argument('--pretrained', type=bool, default=False, help='use pretrained model')
parser.add_argument('--pretrained_path', type=str, default='./model/', help='pretrained model path')
parser.add_argument('--lbfgs', type=bool, default=False, help='use lbfgs')
args = parser.parse_args()


def train_model(model, train_loader, val_loader, epochs_start, loss_start, num_epochs, learning_rate):
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    # criterion = nn.HuberLoss(delta=1)  # HuberLoss delta=1时为平滑MSE
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_loss = loss_start  # 初始化一个损失，用于保存模型时记录最优损失
    element_name = get_element_name(args.element_id)
    # 创建一个 SummaryWriter 实例，指定记录的目录
    writer = SummaryWriter('logs_pixel')
    for epoch in range(num_epochs - epochs_start):  # 从预训练模型的epoch开始训练
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            # 这里要加.to(torch.float32)，要不然报错loss.backward() RuntimeError: Found dtype Double but expected Float
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            loss.backward()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        # 使用 add_scalar 方法记录损失值，第一个参数为标签名，第二个参数为记录的值，第三个参数为当前迭代次数
        writer.add_scalar('train/loss', epoch_loss, epoch + epochs_start + 1)
        print(f"Epoch [{epoch + epochs_start + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss

            loss_train, rmse_train, mae_train, r2_train = validate_model(model, train_loader, criterion)
            loss_val, rmse_val, mae_val, r2_val = validate_model(model, val_loader, criterion)
            print(f'Train_Data:\nLoss:{loss_train} RMSE: {rmse_train} MAE:{mae_train} R2:{r2_train}')
            print(f'Val_Data:\nLoss:{loss_val} RMSE: {rmse_val} MAE:{mae_val} R2:{r2_val}')
            writer.add_scalar('train/rmse', rmse_train, epoch + epochs_start + 1)
            writer.add_scalar('train/mae', mae_train, epoch + epochs_start + 1)
            writer.add_scalar('train/r2', r2_train, epoch + epochs_start + 1)
            writer.add_scalar('val/rmse', rmse_val, epoch + epochs_start + 1)
            writer.add_scalar('val/mae', mae_val, epoch + epochs_start + 1)
            writer.add_scalar('val/r2', r2_val, epoch + epochs_start + 1)
            print(f"Epoch {epoch + epochs_start + 1} save model! Loss={epoch_loss:.4f}")
            torch.save({'model': model.state_dict(), 'epoch': epoch + epochs_start + 1, 'loss': epoch_loss},
                       f"{args.save_model_path}/{element_name}/DNNRNet_best.pth")

        if (epoch + epochs_start + 1) % 10 == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch + epochs_start + 1, 'loss': epoch_loss},
                       f"{args.save_model_path}/{element_name}/DNNRNet_last.pth")
            loss_train, rmse_train, mae_train, r2_train = validate_model(model, train_loader, criterion)
            loss_val, rmse_val, mae_val, r2_val = validate_model(model, val_loader, criterion)
            print(f'Train_Data:\nLoss:{loss_train} RMSE: {rmse_train} MAE:{mae_train} R2:{r2_train}')
            print(f'Val_Data:\nLoss:{loss_val} RMSE: {rmse_val} MAE:{mae_val} R2:{r2_val}')
            writer.add_scalar('train/rmse', rmse_train, epoch + epochs_start + 1)
            writer.add_scalar('train/mae', mae_train, epoch + epochs_start + 1)
            writer.add_scalar('train/r2', r2_train, epoch + epochs_start + 1)
            writer.add_scalar('val/rmse', rmse_val, epoch + epochs_start + 1)
            writer.add_scalar('val/mae', mae_val, epoch + epochs_start + 1)
            writer.add_scalar('val/r2', r2_val, epoch + epochs_start + 1)
        # 记录完数据后关闭 SummaryWriter
        writer.close()


def train_model_LBFGS(model, train_loader, val_loader, epochs_start, loss_start, num_epochs, learning_rate):
    # criterion = nn.MSELoss()  # 使用均方误差损失函数
    criterion = nn.HuberLoss(delta=1)  # HuberLoss delta=1时为平滑MSE
    optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_loss = loss_start  # 初始化一个损失，用于保存模型时记录最优损失
    element_name = get_element_name(args.element_id)

    def closure():
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
        loss.backward()
        return loss

    for epoch in range(num_epochs - epochs_start):  # 从预训练模型的epoch开始训练
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = optimizer.step(closure)
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + epochs_start + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss

            loss_train, rmse_train, mae_train, r2_train = validate_model(model, train_loader, criterion)
            loss_val, rmse_val, mae_val, r2_val = validate_model(model, val_loader, criterion)
            print(f'Train_Data:\nLoss:{loss_train} RMSE: {rmse_train} MAE:{mae_train} R2:{r2_train}')
            print(f'Val_Data:\nLoss:{loss_val} RMSE: {rmse_val} MAE:{mae_val} R2:{r2_val}')

            print(f"Epoch {epoch + epochs_start + 1} save model! Loss={epoch_loss:.4f}")
            torch.save({'model': model.state_dict(), 'epoch': epoch + epochs_start + 1, 'loss': epoch_loss},
                       f"{args.save_model_path}/{element_name}/DNNRNet_best.pth")

        if (epoch + epochs_start + 1) % 10 == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch + epochs_start + 1, 'loss': epoch_loss},
                       f"{args.save_model_path}/{element_name}/DNNRNet_last.pth")
            loss_train, rmse_train, mae_train, r2_train = validate_model(model, train_loader, criterion)
            loss_val, rmse_val, mae_val, r2_val = validate_model(model, val_loader, criterion)
            print(f'Train_Data:\nLoss:{loss_train} RMSE: {rmse_train} MAE:{mae_train} R2:{r2_train}')
            print(f'Val_Data:\nLoss:{loss_val} RMSE: {rmse_val} MAE:{mae_val} R2:{r2_val}')


def search_xy(excel_path, x_y, element_id):  # 寻找x_y位置，构建训练集的y
    excel_data = pd.read_excel(excel_path, sheet_name='Spectra')
    rows = excel_data[(excel_data['X'] == x_y[0]) & (excel_data['Y'] == x_y[1])].index
    # 注意Pandas获取列要使用列名可以写成excel_data['TP_Value'][rows[0]]，或者df.iloc[x,y]获取第x行第y列的值
    y_value = excel_data.iloc[rows, element_id]
    wave_value = excel_data.iloc[rows, 13:229]
    if len(rows) > 0:
        return y_value, wave_value
    else:
        return -1


def to_torchtensor(element_id, img_path):  # 将文件夹中的图片和对应坐标点的excel中的水体参数值，转换为tensor格式
    image_list_train = []
    image_list_val = []
    x_train1 = []
    x_val1 = []
    y_train1 = []
    y_val1 = []

    if element_id == 10:
        img_path_train = img_path + 'TP/train/'
        img_path_val = img_path + 'TP/test/'
    elif element_id == 11:
        img_path_train = img_path + 'TN/train/'
        img_path_val = img_path + 'TN/test/'
    elif element_id == 12:
        img_path_train = img_path + 'NH3/train/'
        img_path_val = img_path + 'NH3/test/'

    for filename in os.listdir(img_path_train):
        str_img_name = os.path.splitext(filename)[0]
        str_img_name = str_img_name.replace('(', '').replace(')', '')
        x, y = map(int, str_img_name.split(','))
        data_tuple = (x, y)
        y_train_temp, x_train_temp = search_xy(args.excel_path, data_tuple, args.element_id)
        x_train1.append(torch.from_numpy(x_train_temp.values).view(216, 1, 1))
        y_train1.append(y_train_temp)

    x_train = torch.stack(x_train1)
    y_train = torch.from_numpy(np.array(y_train1)).reshape(-1, 1)

    for filename in os.listdir(img_path_val):
        str_img_name = os.path.splitext(filename)[0]
        str_img_name = str_img_name.replace('(', '').replace(')', '')
        x, y = map(int, str_img_name.split(','))
        data_tuple = (x, y)
        y_val_temp, x_val_temp = search_xy(args.excel_path, data_tuple, args.element_id)
        x_val1.append(torch.from_numpy(x_val_temp.values).view(216, 1, 1))
        y_val1.append(y_val_temp)

    x_val = torch.stack(x_val1)
    y_val = torch.from_numpy(np.array(y_val1)).reshape(-1, 1)
    return x_train, y_train, x_val, y_val


def validate_model(model_v, val_loader1, criterion):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model_v.eval()
    val_loss = 0.0
    rmse = 0.0
    mae = 0.0
    r2 = 0.0
    total_samples = 0
    # true_y = val_loader1.dataset.tensors[1].cpu().numpy()
    with torch.no_grad():
        for val_inputs, val_labels in val_loader1:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_inputs = val_inputs.to(torch.float32)
            val_labels = val_labels.to(torch.float32)
            val_outputs = model_v(val_inputs)
            total_samples += val_loader1.batch_size
            val_loss += criterion(val_outputs.to(torch.float32), val_labels.to(torch.float32)).item() * val_inputs.size(
                0)
            predict_y = val_outputs.cpu().numpy()
            true_y = val_labels.cpu().numpy()
            rmse = rmse + mean_squared_error(true_y, predict_y) * val_loader1.batch_size
            mae = mae + mean_absolute_error(true_y, predict_y) * val_loader1.batch_size
            r2 = r2 + r2_score(true_y, predict_y) * val_loader1.batch_size
    avg_val_loss = val_loss / total_samples
    avg_val_rmse = rmse / total_samples
    avg_val_mae = mae / total_samples
    avg_val_r2 = r2 / total_samples
    return avg_val_loss, avg_val_rmse, avg_val_mae, avg_val_r2


def get_element_name(element_id):  # 获取元素名的函数
    if element_id == 10:
        element_name = 'TP'
    elif element_id == 11:
        element_name = 'TN'
    elif element_id == 12:
        element_name = 'NH3'
    return element_name


x_train, y_train, x_val, y_val = to_torchtensor(args.element_id, args.train_data)

# 转换为 PyTorch 的 Dataset
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

# 创建 DataLoader 加载数据
batch_size = args.train_batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

for data in train_loader:
    inputs, labels = data

# 创建模型实例
element_name = get_element_name(args.element_id)
model = DNNRNet_point.DnnrNet(in_channels=args.in_channels)
epochs_start = 0
loss_start = 0.05
if args.pretrained:
    state_dict = torch.load(f'{args.pretrained_path}{element_name}/DNNRNet_Best.pth')
    model.load_state_dict(state_dict['model'])
    epochs_start = state_dict['epoch']
    loss_start = state_dict['loss']

# 调用训练函数开始训练
if args.lbfgs:
    train_model_LBFGS(model, train_loader, val_loader, epochs_start, loss_start, args.num_epochs, args.learning_rate)
else:
    train_model(model, train_loader, val_loader, epochs_start, loss_start, args.num_epochs, args.learning_rate)
