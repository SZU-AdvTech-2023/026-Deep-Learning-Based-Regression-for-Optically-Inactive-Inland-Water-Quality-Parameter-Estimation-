import torch
import argparse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import train
import DNNRNet

parser = argparse.ArgumentParser(description='DNNRNet_args')
parser.add_argument('--test_batch_size', type=int, default=10, help='test_batch_size')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for test')
parser.add_argument('--in_channels', type=int, default=57, help='Number of input channels')
parser.add_argument('--test_data', type=str, default='', help='Train data Loc')
parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu_id')
parser.add_argument('--element_id', type=int, default=10, help='TP:10,TN:11,NH3:12')
parser.add_argument('--excel_path', type=str, default='')
parser.add_argument('--save_model_path', type=str, default='./model', help='path to save model')
parser.add_argument('--pretrained_path', type=str, default='./model/', help='pretrained model path')
parser.add_argument('--lbfgs', type=bool, default=False, help='use lbfgs')
args = parser.parse_args()


def validate_model(model_v, val_loader1, criterion):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model_v.to(device)
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


if __name__ == '__main__':
    x_train, y_train, x_val, y_val = train.to_torchtensor(args.element_id, args.test_data)
    # 转换为 PyTorch 的 Dataset
    val_dataset = TensorDataset(x_val, y_val)

    model = DNNRNet.DnnrNet(in_channels=args.in_channels)
    state_dict = torch.load(f'{args.pretrained_path}TP/DNNRNet_best.pth')
    model.load_state_dict(state_dict['model'])
    epochs_start = state_dict['epoch']
    loss_start = state_dict['loss']
    criterion = nn.MSELoss()
    # 创建 DataLoader 加载数据
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
    print(validate_model(model, val_loader, criterion))
