from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

# 准备数据
x=np.linspace(-np.pi,np.pi,100)
#x=np.random.uniform(0,2,400)
y=np.sin(x)
# 将数据做成数据集的模样
X=np.expand_dims(x,axis=1)
Y=y.reshape(100,-1)
# 使用批训练方式
dataset=TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=10,shuffle=True)

# 神经网络主要结构，这里就是一个简单的线性结构

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1=nn.Linear(1,10)
    self.fc2=nn.Linear(10,1)
    #  nn.Linear(10,100),nn.ReLU(),
    #  nn.Linear(100,10),nn.ReLU(),
    #  nn.Linear(3,1),
    # nn.Sigmoid()
    for m in self.modules():
      if isinstance(m, nn.Linear):
        m.weight.data = torch.nn.init.normal_(m.weight.data,std=0.6)
        m.bias.data = torch.nn.init.normal_(m.bias.data,std=0.01)
  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x

net=Net()
# params = list(net.parameters())
#
# print(params[0].data)
# w10=torch.nn.init.normal_(params[0].data)
# b1=torch.nn.init.normal_(params[1].data)
# w21=torch.nn.init.normal_(params[2].data)
#print(Net.parameters(net))
#print(Net.parameters(net))
#定义优化器和损失函数
#print(layer.weight.data)

optim=torch.optim.Adam(Net.parameters(net),lr=0.001)
Loss=nn.MSELoss()

# 下面开始训练：
# 一共训练 1000次
for epoch in range(100):
  loss=None
  for batch_x,batch_y in dataloader:
    #print(batch_x)
    y_predict=net(batch_x)
    loss=Loss(y_predict,batch_y)
    #print(loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
  # 每100次 的时候打印一次日志
  if (epoch+1)%5==0:
    print("step: {0} , loss: {1}".format(epoch+1,loss.item()))

# 使用训练好的模型进行预测
predict=net(torch.tensor(X,dtype=torch.float))

# 绘图展示预测的和真实数据之间的差异
import matplotlib.pyplot as plt
plt.plot(x,y,label="fact")
plt.plot(x,predict.detach().numpy(),label="predict")
plt.title("sin function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()
#plt.savefig(fname="result_square7.png",figsize=[10,10])