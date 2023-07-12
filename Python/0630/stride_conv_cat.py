import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

image_path = "./data/cat.jpg"
image = Image.open(image_path).convert('L')
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이미지를 텐서로 변환
input_data = torch.unsqueeze(torch.from_numpy(np.array(image)), dim=0).float()
input_data = torch.unsqueeze(input_data,dim=0)
# print(input_data.shape)

# Convolution layer 생성(stride=2)
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
output = conv(input_data)

plt.subplot(1,2,1)
plt.imshow(input_data.squeeze().detach().numpy(), cmap='gray')
plt.title('input image')

plt.subplot(1,2,2)
plt.imshow(output.squeeze().detach().numpy(), cmap='gray')
plt.title('output image')

plt.tight_layout()
plt.show()