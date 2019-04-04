# 统计学知识梳理

##  统计学基本知识

### statistics 基本知识
**descriptive**
	描述数据部分
**inferential**
	预测数据部分

#### Ⅰ mean median mode
**mean** 平均数：
	$\overline{x}=\frac{\sum_{i=1}^{n}x_i}{n}​$
**median** 中位数：在中间出现的数
**mode**众数：出现最多的数

#### Ⅱ range midrange
**range** 极差：
	$range(x)=max(x_i)-min(x_i)$
**midrange**中程数：
	$midrange(x)=\frac{max(x_i)+min(x_i)}{2}$

#### Ⅲ 集中趋势度量
**define**:  the measure of a set of  numbers
主要的衡量指标包括：
Mean(Arithmatic)：平均值
median：中位数
mode:the mose commen number 
outlier:a number that really kind of sticks out
numbers：3 3 3 3 3 100

#### Ⅳ 分散度度量
**mean:**
sample and population
抽样的关键是random
sample mean : $\overline{x}$
population mean: $\mu$

##### 离散性
**全距数：**
	$range(x)=max(x_i)-min(x_i)$
**四分位距: ** 
	四分位数 Q1,Q2,Q3
	$range(x)_{\frac{1}{4}}=max(x_i)-min(x_i)$
**百分位数**
	同四分位距，将数分为100份
*`PS.可以用箱线图绘制各种距`*
##### 变异性
**方差:**
population variance:
	$\sigma^{2}=\frac{\sum_{i=1}^{N}(x_i-\mu)^2}{N}=\frac{\sum_{i=1}^{n}x_i^2}{n}-\mu^2$
sample variance:
	$S^{2}=\frac{\sum_{i=1}^{n}(x_i-\overline{x})^2}{n}$
the better estimate of the population variance:
	$S_{n-1}^{2}=\frac{\sum_{i=1}^{n}(x_i-\overline{x})^2}{n-1}$

**标准差：**
	$\sigma=\sqrt\frac{\sum_{i=1}^{N}(x_i-\mu)^2}{N}=\sqrt{\frac{\sum_{i=1}^{n}x_i^2}{n}-\mu^2}​$

### 概率计算
	$P(A)=\frac{n(A)}{n(S)}​$
*A位可能发生的概率*
*S被称为概率空间，或者样本空间*
*`PS.韦恩图可以概率的图形`*

#### Ⅰ 事件
**对立事件**
	$P(A')=1-P(A)$
**互斥事件**
**相交事件**
	$P({A}\cup{B})=P(A)+P(B)-P({A}\cap{B})$
*比如：黑色，又是偶数的球*
#### Ⅱ 条件概率
B发生条件下，A发生的概率：A和B同时发生的概率/B发生的概率
	$P(A|B)=\frac{P({A}\cap{B})}{P(B)}$
改变公式，就有：
	$P({A}\cap{B})=P(A|B)\times P(B)$
	$P({A}\cap{B})=P(B|A)\times P(A)​$
*`PS.概率树可以计算条件概率`*

#### Ⅲ 贝叶斯定理
贝叶斯公式：
	$P(A|B)=\frac{P(B|A)\times P(A)}{P(B|A)\times P(A)+P(B|A')\times P(A')}$
全概率公式：
	$P(B)=P(B|A)\times P(A)+P(B|A')\times P(A')$
练习题：
有两种新游戏，一群志愿者试玩。80%的志愿者选择了1，20%的志愿者选择了2。在游戏1中，60%的人觉得好玩，40%的人觉得不好玩。在游戏2中，70%的人觉得好玩，30%的人觉得不好玩。
现在随机挑选一个志愿者，他认为游戏好玩时，这款游戏是2的概率有多大。
假设游戏为A,B，好玩与否为C,D
	$P(B|D)=\frac{P(D|B)\times P(B)}{P(D|B)\times P(B)+P(D|A)\times P(A)}​$
他认为游戏好玩时，这款游戏是2的概率为：选择2并且认为好玩的概率除以所有认为好玩的概率

**相关事件与独立事件**
事件之间相互影响，则称为相关事件
事件之间互不影响，则称为独立事件
若两个事件是独立事件：
	$P(A|B)=\frac{P({A}\cap{B})}{P(B)}$ ==> $P(A)=\frac{P({A}\cap{B})}{P(B)}$
	则有$P({A}\cap{B})=P(A)\times P(B)$
贝叶斯定理，独立事件：
	$P(A|B)=\frac{P(B)\times P(A)}{P(B|A)\times P(A)+P(B|A')\times P(A')}$

### 离散概率分布的期望与方差

#### 期望，方差，标准差
**期望：**
	每个数值乘以该数值的发生概率
	$E(X)=\sum{x}P(X=x)$
	$E(f(x))=\sum{f(x)}{P(X=x)}$
**方差：**
	$Var(X)=E(X-\mu)^2=E(X^2)-\mu^2$
**标准差：**
	$\sigma=\sqrt{var(X)}$
**线性变换的通用公式：**
	$E(aX+b)=aE(X)+b$
	$Var(ax+b)=a^2Var(X)$
	$Var(ax+b)=a^2Var(X)$
	$E(X^2)=\sum x^2P(X=x)$
	$Var(aX-b)=a^2Var(x)​$

### 排列组合
**排列：**
	排列指从一个较大(n个)对象群体中取出一定数目(r个)进行排序，并得出排序方式总数目。
	$P^n_r=\frac{n!}{(n-r)!}$
**组合：**
	组合是指从一个群体中选取几个对象，在不考虑这几个对象的顺序的情况下，求出这几个对象的选取方式的数目。
	$C^n_r=\frac{n!}{r!(n-r)!}$

## 概率分布

### 二项分布

### 	

​	