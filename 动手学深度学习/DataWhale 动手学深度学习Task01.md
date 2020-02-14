# DataWhale 动手学深度学习（一）

注：课程来自于伯禹平台：<https://www.boyuai.com/elites/>

学习笔记，记录主要知识点和课后习题。

### 预备知识

#### 人工神经网络

##### 简介

神经网络组成：

​	真正的神经元：细胞体，树突，轴突，突触末梢

​    神经通信：

​			1.细胞膜间的电位表现出的电信号称为动作电位

​			2.电信号从细胞体中产生，沿着轴突往下转，并导致突触末梢释放神经递质介质

​			3.介质通过化学扩散从突出传递到其他神经元的树突

​			4.神经递质可以试兴奋的或者是抑制的

​			5.如果从其他神经元来的神经递质是兴奋的并且超过了某个阈值，他就会触发一个运动电位

##### McCulloch-Pitts神经元模型建立：

- 将网络建模成一个图，单元作为节点，突触连接作为从节点i到节点j的加权边，权重为$w_{ji}$

- 将单元的网络输入建模为公式
  				
  $$
  net_j=\sum_i{w_{ji}o_i}
  $$

- 单元输出为
  $$
  o_j=
  \begin{cases}
  0 & if & net_j<T_j \\
  1 & if & net_j \ge T_j
  \end{cases}
  $$
  $T_j$是单元j的阈值

##### 感知机模型：

输入${x_1,x_2...x_m}$，权重$w_1,w_2...w_m$,Bias $b$ ,激活函数$\varphi()$

![感知机模型.png](https://github.com/SpenserZC/DataWhale/blob/master/TyporaPictureRepo/deeplearning/%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%A8%A1%E5%9E%8B.png?raw=true)

- Rosenblatt进一步提出感知机作为第一个在“老师”指导下进行学习的模型（监督学习）

- 专注如何找到合适的用于而分类任务的权重$w_m$

- 训练

  
  $$
  w_i = w_i +\eta(y-\hat{y})x_i \\
  b = b+\eta(y-\hat{y})
  $$

- 下列规则等价：

  - 如果输出正确，则不进行操作

  - 如果输出高了，降低正输入的权重

  - 如果输出低了，增加正输入的权重

    

Rosenblatt证明了学习算法的收敛性，如果两个类别是线性可分的。



##### 多层感知机：

###### 隐藏层和反向传播：

前馈

- 消息从输入节点触发，经过隐藏节点（如果有的话）到输出节点
- 网络中没有圈或者循环

![双层感知机多层计算.png](https://github.com/SpenserZC/DataWhale/blob/master/TyporaPictureRepo/deeplearning/%E5%8F%8C%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA%E5%A4%9A%E5%B1%82%E8%AE%A1%E7%AE%97.png?raw=true)

###### 常用激活函数

Sigmoid激活函数：
$$
\sigma(z) =\frac{1}{1+e^{-z}}
$$
Tanh激活函数：
$$
tanh(z) = \frac{1-e^{-2z}}{1+e^{-2z}}
$$
线性整流函数（Rectified Linear Unit, ReLU）:
$$
ReLU(Z) = max(0,z)
$$


卷积神经网络：

循环神经网络：





### 线性回归

#### 线性回归基本要素

##### 模型

为了简单起见，这里我们假设价格只取决于房屋状况的两个因素，即面积（平方米）和房龄（年）。接下来我们希望探索价格与这两个因素的具体关系。线性回归假设输出与各个输入之间是线性关系:
$$
price=w_{area}⋅area+w_{age}⋅age+b
$$
注：线性回归的模型最后就是一个线性方程。

##### 数据集

我们通常收集一系列的真实数据，例如多栋房屋的真实售出价格和它们对应的面积和房龄。我们希望在这个数据上面寻找模型参数来使模型的预测价格与真实价格的误差最小。在机器学习术语里，该数据集被称为训练数据集（training data set）或训练集（training set），一栋房屋被称为一个样本（sample），其真实售出价格叫作标签（label），用来预测标签的两个因素叫作特征（feature）。特征用来表征样本的特点。

##### 损失函数

在模型训练中，我们需要衡量价格预测值与真实值之间的误差。通常我们会选取一个非负数作为误差，且数值越小表示误差越小。一个常用的选择是平方函数。 它在评估索引为 i 的样本误差的表达式为
$$
l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2, \\
L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.
$$

##### 优化函数

当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）。本节使用的线性回归和平方误差刚好属于这个范畴。然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）。

在求数值解的优化算法中，小批量随机梯度下降（mini-batch stochastic gradient descent）在深度学习中被广泛使用。它的算法很简单：先选取一组模型参数的初始值，如随机选取；接下来对参数进行多次迭代，使每次迭代都可能降低损失函数的值。在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）B，然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。
$$
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)
$$

##### 矢量计算

在模型训练或预测时，我们常常会同时处理多个数据样本并用到矢量计算。在介绍线性回归的矢量计算表达式之前，让我们先考虑对两个向量相加的两种方法。

1. 向量相加的一种方法是，将这两个向量按元素逐一做标量加法。
2. 向量相加的另一种方法是，将这两个向量直接做矢量加法。

##### 利用pytorch实现线性回归模型

```python
import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')

# 生成数据集
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据集
import torch.utils.data as Data

batch_size = 10

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # 每次取得数据批量大小
    shuffle=True,               # 是否随机取出
    num_workers=2,              # 取出的线程数
)

for X, y in data_iter:
    print(X, '\n', y)
    break
    
# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y
# 神经网络，现在不太清楚，这块生成单层网络。下个代码块生成多个网络的方法    
net = LinearNet(num_inputs)
print(net)

# 初始化模型参数
from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly
for param in net.parameters():
    print(param)
    
# 定义损失函数
loss = nn.MSELoss()

# 定义优化函数
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)   # built-in random gradient descent function
print(optimizer)

# 训练
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
    
# 结果
dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)
```

```python
# 生成神经网络的方法
# ways to init a multilayer network
# method one
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )

# method two
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])
```

### softmax与分类模型

#### softmax的基本概念

###### 分类问题

一个简单的图像分类问题，输入图像的高和宽均为2像素，色彩为灰度。
图像中的4像素分别记为$x_1,x_2,x_3,x_4。$

假设真实标签为狗、猫或者鸡，这些标签对应的离散值为$y_1,y_2,y_3。$

我们通常使用离散的数值来表示类别，例如$y_1=1, y_2=2, y_3=3$

###### 权重矢量

$$
\begin{aligned} o_1 &= x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1 \end{aligned} \\
\begin{aligned} o_2 &= x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2 \end{aligned} \\
\begin{aligned} o_3 &= x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3 \end{aligned}
$$

###### 神经网络图

下图用神经网络图描绘了上面的计算。softmax回归同线性回归一样，也是一个单层神经网络。由于每个输出$o1,o2,o3$的计算都要依赖于所有的输入$x1,x2,x3,x4$，softmax回归的输出层也是一个全连接层。

![Image Name](https://cdn.kesci.com/upload/image/q5hmymezog.png)

既然分类问题需要得到离散的预测输出，一个简单的办法是将输出值$oi$当作预测类别是ii的置信度，并将值最大的输出所对应的类作为预测输出，即输出 $argmaxioi$。例如，如果$o1,o2,o3$分别为$0.1,10,0.1$，由于$o2$最大，那么预测类别为2，其代表猫。

###### 输出问题

直接使用输出层的输出有两个问题：

1. 一方面，由于输出层的输出值的范围不确定，我们难以直观上判断这些值的意义。例如，刚才举的例子中的输出值10表示“很置信”图像类别为猫，因为该输出值是其他两类的输出值的100倍。但如果$o_1=o_3=10^3$，那么输出值10却又表示图像类别为猫的概率很低。
2. 另一方面，由于真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量。

softmax运算符（softmax operator）解决了以上两个问题。它通过下式将输出值变换成值为正且和为1的概率分布：
$$
\hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{softmax}(o_1, o_2, o_3)
$$

$$
\hat{y}1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\quad \hat{y}2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\quad \hat{y}3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
$$

容易看出$\hat{y}_1+\hat{y}_2+\hat{y}_3=1$且$0 \leq \hat{y}_1, \hat{y}_2, \hat{y}_3 \leq 1$，因此$\hat{y}_1, \hat{y}_2, \hat{y}_3$是一个合法的概率分布。这时候，如果$\hat{y}_2=0.8$，不管$\hat{y}_1$和$\hat{y}_3$的值是多少，我们都知道图像类别为猫的概率是80%。此外，我们注意到
$$
\underset{i}{\arg\max} o_i = \underset{i}{\arg\max} \hat{y}_i
$$
因此softmax运算不改变预测类别输出。

- 计算效率

  - 单样本矢量计算表达式
    为了提高计算效率，我们可以将单样本分类通过矢量计算来表达。在上面的图像分类问题中，假设softmax回归的权重和偏差参数分别为

  - $$
    \boldsymbol{W} = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\ w_{31} & w_{32} & w_{33} \\ w_{41} & w_{42} & w_{43} \end{bmatrix},\quad \boldsymbol{b} = \begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix},设高和宽分别为2个像素的图像样本i的特征为
    $$

    设高和宽分别为2个像素的图像样本i的特征为

设高和宽分别为2个像素的图像样本i的特征为
$$
\boldsymbol{x}^{(i)} = \begin{bmatrix}x_1^{(i)} & x_2^{(i)} & x_3^{(i)} & x_4^{(i)}\end{bmatrix},
$$
输出层的输出为
$$
\boldsymbol{o}^{(i)} = \begin{bmatrix}o_1^{(i)} & o_2^{(i)} & o_3^{(i)}\end{bmatrix},
$$
预测为狗、猫或鸡的概率分布为
$$
\boldsymbol{\hat{y}}^{(i)} = \begin{bmatrix}\hat{y}_1^{(i)} & \hat{y}_2^{(i)} & \hat{y}_3^{(i)}\end{bmatrix}.
$$
softmax回归对样本i分类的矢量计算表达式为
$$
\begin{aligned} \boldsymbol{o}^{(i)} &= \boldsymbol{x}^{(i)} \boldsymbol{W} + \boldsymbol{b},\\ \boldsymbol{\hat{y}}^{(i)} &= \text{softmax}(\boldsymbol{o}^{(i)}). \end{aligned}
$$
小批量矢量计算表达式:
为了进一步提升计算效率，我们通常对小批量数据做矢量计算。广义上讲，给定一个小批量样本，其批量大小为n，输入个数（特征数）为d，输出个数（类别数）为q。设批量特征为$\boldsymbol{X} \in \mathbb{R}^{n \times d}$。假设softmax回归的权重和偏差参数分别为$\boldsymbol{W} \in \mathbb{R}^{d \times q}$和$\boldsymbol{b} \in \mathbb{R}^{1 \times q}$.softmax回归的矢量计算表达式为
$$
\begin{aligned} \boldsymbol{O} &= \boldsymbol{X} \boldsymbol{W} + \boldsymbol{b},\\ \boldsymbol{\hat{Y}} &= \text{softmax}(\boldsymbol{O}), \end{aligned}
$$
其中的加法运算使用了广播机制，$\boldsymbol{O}, \boldsymbol{\hat{Y}} \in \mathbb{R}^{n \times q}$,且这两个矩阵的第$i$行分别为样本i的输出$o^{(i)}$和概率分布$y^{(i)}$.

##### 交叉熵损失函数

对于样本i，我们构造向量$\boldsymbol{y}^{(i)}\in \mathbb{R}^{q}$，使其第$y^{(i)}$（样本$i$类别的离散数值）个元素为1，其余为0。这样我们的训练目标可以设为使预测概率分布$y^{(i)}$尽可能接近真实的标签概率分布$y^{(i)}$。

- 平方损失估计

$$
\begin{aligned}Loss = |\boldsymbol{\hat y}^{(i)}-\boldsymbol{y}^{(i)}|^2/2\end{aligned}
$$

然而，想要预测分类结果正确，我们其实并不需要预测概率完全等于标签概率。例如，在图像分类的例子里，如果$y^{(i)}=3$，那么我们只需要$\hat{y}^{(i)}_3$比其他两个预测值$\hat{y}^{(i)}_1$和$\hat{y}^{(i)}_2$大就行了。即使$\hat{y}^{(i)}_3$值为0.6，不管其他两个预测值为多少，类别预测均正确。而平方损失则过于严格，例如$\hat y^{(i)}_1=\hat y^{(i)}_2=0.2$比$\hat y^{(i)}_1=0, \hat y^{(i)}_2=0.4$的损失要小很多，虽然两者都有同样正确的分类预测结果。

改善上述问题的一个方法是使用更适合衡量两个概率分布差异的测量函数。其中，交叉熵（cross entropy）是一个常用的衡量方法：
$$
H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},
$$
其中带下标的$y_j^{(i)}$是向量$\boldsymbol y^{(i)}$中非0即1的元素，需要注意将它与样本i类别的离散数值，即不带下标的$y^{(i)}$区分。在上式中，我们知道向量$\boldsymbol y^{(i)}$中只有第$y^{(i)}$个元素$y^{(i)}{y^{(i)}}$为1，其余全为0，于是$H(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}) = -\log \hat y{y^{(i)}}^{(i)}$。也就是说，交叉熵只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。当然，遇到一个样本有多个标签时，例如图像里含有不止一个物体时，我们并不能做这一步简化。但即便对于这种情况，交叉熵同样只关心对图像中出现的物体类别的预测概率。

假设训练数据集的样本数为n，交叉熵损失函数定义为
$$
\ell(\boldsymbol{\Theta}) = \frac{1}{n} \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),
$$
其中$\boldsymbol{\Theta}$代表模型参数。同样地，如果每个样本只有一个标签，那么交叉熵损失可以简写成$\ell(\boldsymbol{\Theta}) = -(1/n) \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$。从另一个角度来看，我们知道最小化ℓ(Θ)等价于最大化$\exp(-n\ell(\boldsymbol{\Theta}))=\prod_{i=1}^n \hat y_{y^{(i)}}^{(i)}$，即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

### 多层感知机

##### 基本知识

###### 隐藏层

下图展示了一个多层感知机的神经网络图，它含有一个隐藏层，该层中有5个隐藏单元。



![Image Name](https://cdn.kesci.com/upload/image/q5ho684jmh.png)

###### 表达式

给定一个小批量样本$\boldsymbol{X} \in \mathbb{R}^{n \times d}$，其批量大小为n，输入个数（特征数）为d，输出个数（类别数）为q。记隐藏层的输出（也称为隐藏层变量或隐藏变量）为 $\boldsymbol{H}$,有$ \boldsymbol{H} \in \mathbb{R}^{n \times h}$。因为隐藏层和输出层均是全连接层，可以设隐藏层的权重参数和偏差参数分别为$\boldsymbol{W_h} \in \mathbb{R}^{d \times h}$和$\boldsymbol{b_h} \in \mathbb{R}^{1 \times h}$.输出层的权重和偏差参数分别为$\boldsymbol{W}_o \in \mathbb{R}^{h \times h}$和$\boldsymbol{b}_o \in \mathbb{R}^{1 \times h}$.

我们先来看一种含单隐藏层的多层感知机的设计。其输出$\boldsymbol{O} \in \mathbb{R}^{n \times q}$的计算为：
$$
\begin{aligned} \boldsymbol{H} &= \boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h,\\ \boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o, \end{aligned}
$$
也就是将隐藏层的输出直接作为输出层的输入。如果将以上两个式子联立起来，可以得到
$$
\boldsymbol{O} = (\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h)\boldsymbol{W}_o + \boldsymbol{b}_o = \boldsymbol{X} \boldsymbol{W}_h\boldsymbol{W}_o + \boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o.
$$
从联立后的式子可以看出，虽然神经网络引入了隐藏层，却依然等价于一个单层神经网络：其中输出层权重参数为$\boldsymbol{W}_h\boldsymbol{W}_o$，偏差参数为$\boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o$。不难发现，即便再添加更多的隐藏层，以上设计依然只能与仅含输出层的单层神经网络等价。

