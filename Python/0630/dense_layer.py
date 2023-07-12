import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 입력 크기 4, 출력 크기 2 (임의 지정)
input_size = 4
output_size = 2

# 밀집층 정의
dense_layer = nn.Linear(input_size, output_size)

weights = dense_layer.weight.detach().numpy()

plt.imshow(weights, cmap='coolwarm', aspect='auto')
plt.xlabel("Input Features") # 입력 요소 4
plt.ylabel("Output Units") # 출력 요소 2
plt.title("Dense Layer Weights")
plt.colorbar()
plt.show()