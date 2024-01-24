import os
import imageio.v3 as imageio
from scipy.ndimage import rotate
import numpy as np

# 输入文件夹和输出文件夹路径
input_folder = ''
output_folder = ''
# 遍历文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        # 读取图像
        img_path = os.path.join(input_folder, filename)
        img = imageio.imread(img_path)

        # 生成45度到315度的所有角度并旋转
        for angle in range(0, 271, 90):
            rotated_image = rotate(img, angle, axes=(1, 2), reshape=False)

            # 生成输出文件名（可以根据需要修改文件名）
            output_filename = f'{os.path.splitext(filename)[0]}_{angle}deg.tif'
            output_path = os.path.join(output_folder, output_filename)

            # 保存旋转后的图像
            imageio.imwrite(output_path, rotated_image)
