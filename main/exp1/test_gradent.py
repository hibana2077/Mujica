import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 1. 定義簡單的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3, 1)  # 單層線性模型

    def forward(self, x):
        return self.fc(x)

# 2. 準備數據集
X = torch.randn(100, 3)  # 隨機生成數據 (100 個樣本，2 個特徵)
y = 23 * X[:, 0] + 45 * X[:, 1] + 67 * X[:, 2]
X_noise = torch.randn(10, 3)  # 隨機生成噪音
y_noise = 0.1 * torch.randn(10)  # 隨機生成噪音
X = torch.cat([X, X_noise], dim=0)
y = torch.cat([y, y_noise], dim=0)
dataset = TensorDataset(X, y) # last 10 samples are noise
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 3. 初始化模型、損失函數和優化器
model = SimpleModel()
criterion = nn.MSELoss()  # 均方誤差損失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 收集梯度影響力
epoch_gradient_influence = []

for epoch in range(10):  # 訓練 5 個 epoch
    gradient_influence = []
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, targets)  # 計算損失
        
        loss.backward()  # 反向傳播
        
        # 收集每個樣本的梯度影響力
        
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                # 儲存整個參數的梯度
                gradients.append(param.grad.detach().clone())
        flatten_gradients = torch.cat([grad.flatten() for grad in gradients])
        gradient_influence.append(flatten_gradients)
        optimizer.step()  # 更新參數

        print(f"                                                     ", end="\r")
        print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}", end="\r")

    epoch_gradient_influence.append(gradient_influence)

# 5. 分析梯度影響力
print(f"                                                     ", end="\r")
print(f"Final Loss: {loss.item()}")
print("Collected Gradient Influence:")
# for sample, gradients in gradient_influence[:2048:512]:
#     print(f"Sample: {sample}")
#     for grad in gradients:
#         print(f"Gradient: {grad}")

for epoch, gradient_influence in enumerate(epoch_gradient_influence):
    print(f"Epoch {epoch + 1}:")
    for gradients in gradient_influence[::110]:
        print(f"Gradients: {gradients}")

# to numpy
epoch_gradient_influence_np = np.array(epoch_gradient_influence)
print(epoch_gradient_influence_np.shape)

diff = np.abs(np.diff(epoch_gradient_influence_np, axis=0))
diff = diff.mean(axis=2)
diff_mean = np.mean(diff, axis=0)
diff_var = np.var(diff, axis=0)
print(diff.shape)
print(diff_mean.shape)
print(diff_var.shape)

non_noise_diff_mean = diff_mean[:-10]
non_noise_diff_var = diff_var[:-10]

noise_diff_mean = diff_mean[-10:]
noise_diff_var = diff_var[-10:]

# dot plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(non_noise_diff_mean, non_noise_diff_var, label="Non-noise samples")
plt.scatter(noise_diff_mean, noise_diff_var, label="Noise samples")
plt.legend()
plt.xlabel("Mean of Gradient Influence Difference")
plt.ylabel("Variance of Gradient Influence Difference")
plt.title("Dot Plot of Gradient Influence Difference")
plt.show()