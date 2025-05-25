# 利用MAML进行回归预测
import torch
import learn2learn

# 数据集导入部分
x = torch.tensor([[1.0],[2.0],[3.0]])
y= torch.tensor([[2.0],[4.0],[6.0]])

# 线性模型类
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()


# 超参数设置
# 定义maml的内循环和外循环学习率
meta_lr = 0.005
fast_lr = 0.05

# 建立MAML模型
maml_qiao = learn2learn.algorithms.MAML(model, lr=fast_lr)

# 定义优化器
opt = torch.optim.Adam(maml_qiao.parameters(), meta_lr)

# 定义损失函数
loss = torch.nn.MSELoss()

#开始训练
for epoch in range(100):
    clone = maml_qiao.clone()
    #进行预测
    y_pred=clone(x)
    error = loss(y_pred, y)
    print(epoch, error)
    clone.adapt(error)
    opt.zero_grad()
    error.backward()
    opt.step()

