import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 입력 크기 10, 출력 크기 4 (임의 지정)
input_size = 10
output_size = 4

# 밀집층 정의
output_layer = nn.Linear(input_size, output_size)

weights = output_layer.weight.detach().numpy()

plt.imshow(weights, cmap='coolwarm', aspect='auto')
plt.xlabel("Input Features") # 입력 요소 4
plt.ylabel("Output Units") # 출력 요소 2
plt.title("Dense Layer Weights")
plt.colorbar()
plt.show()