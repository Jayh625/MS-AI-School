import torch
import torch.nn as nn
import matplotlib.pyplot as plt

input_data = torch.randn(1,1,28,28) # batch_size 1, 단일 채널, 28 * 28 크기
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
output = conv(input_data)
# Convolution layer 적용
print(output.shape)

# 입력 데이터 시각화
plt.subplot(1,2,1)
plt.imshow(input_data.squeeze(), cmap='gray')
plt.title('input') 

# 출력 데이터 시각화
plt.subplot(1,2,2)
plt.imshow(output.squeeze().detach().numpy()[0], cmap='gray')
plt.title('output')

plt.tight_layout()
plt.show()