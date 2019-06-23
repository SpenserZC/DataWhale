6月22日

# Python for Data Analysis 学习笔记

### 第一章

#### 1.1本书的内容

本书主要讲的是利用Python进行数据控制，处理，整理，分析等方面的具体细节和基本要点。目标是介绍Python编程和应用数据处理的库和工具环境。

##### 什么样的数据？

结构化数据：表格型数据；多维数组；通过关键列相互联系的多个表；间隔平均或不平均的时间序列。大部分数据集都能不转换为更适合分析和建模的结构化形式。例如，一组新闻文章可以被处理为一张词频表，而这张词频表就可以用于情感分析。

#### 1.2Why Python

1.2.1Python 作为胶水语言。可以以轻松集成C，C++以及Fortran库。

1.2.2解决“两种语言问题”。通常使用SAS和R对新的算法，原型进行构建和测试，然后再移植到大的生产系统中去（JAVA，C#或C++）。python可以做模型，也可以构建系统。

1.2.3缺点：解释型语言，比编译型语言慢的多；高并发，多线程不适用（全局解释器锁）。

#### 1.3重要的Python库

###### Numpy

Python 科学计算的基础包。数组，读写数据集，线性代数运算等等。

###### pandas

快速便捷处理结构化数据的大量数据结构和函数提供复杂精细的索引功能，很便捷地完成重塑，切片和切块，聚合以及选取数据子集等操作。

###### matplotlib

绘制图表和其它二维数据可视化的Python库。

###### Ipython和Jupyter

Ipython shell和Jupyter notebooks特别适合进行数据探索和可视化。

###### SciPy

一组专门解决科学计算中各种标准问题域的包的集合。包括数值积分方程和微分方程求解器；先行底数和矩阵分解功能；函数优化器（最小化器）以及根查找算法。

###### scikit-learn

机器学习通用包

###### statsmodels

统计分析包。

#### 1.4安装和设置

anaconda安装

### 第二章Python语法基础，IPython和Jupyter Notebooks

#### 2.1Python 解释器

python是解释性语言。所以需要解释器。

#### 2.2 IPython和Jupyter Notebooks使用

###### tab补全

自动补全已输入的变量（对象，函数等等）的命名空间

###### 自省

变量前后使用问好？,显示对象的信息

函数之后使用？？,显示函数的源码

？还可以所有IPython的命名空间。

###### %run

你可以使用%run命令运行所有的Python程序。

###### 终端运行代码

Ctrl+C

###### 从剪切板执行程序

%past 执行剪切板的程序，%cpast 执行剪切板任意行的数据。

###### 魔术命令

Ipython中特殊的命令。在指令前添加百分号%前缀。例如：%timeit测量任何Python语句的执行时间。

###### 集成Matplotlib

%matplotlib魔术函数配置了Ipython shell和Jupyter notebook中的matplotlib

#### 2.3Python 语法基础

略~

### 第三章 Python的数据结构、函数和文件

#### 3.1数据结构和序列

###### 元组

是一个固定长度，不可改变的Python序列对象。创建元组的最简单方式，是用逗号分隔一列值。

用tuple可以将任意序列或迭代器转换成元组。

**拆分元组**：可以使用*rest语法，从函数签名中抓取任意长度列表的位置参数。

tuple方法：

In [34]:  a=(1,2,2,2,2,4,5)

In [35]:  a.conut(2) 

out[35]: 4

###### 列表

与元组对比。列表的长度可变、内容可以被修改。你可以用方括号定义，或用list函数：a_list=[2,3,7,None]

1. 添加或删除元素

   ​	在列表尾添加：list.append() 

   ​	在特定位置添加：list.insert(1,'red')

   ​    移除并返回指定位置的元素：list.pop(2)

   ​	remove移除，会先寻找第一个值并除去：list.remove('red')

   ​	检查是否包含，in，not in：‘red’ in list   'red' not in list

2. 串联和组合列表

   ​	加号+，可以串联两个列表

   ​	extend,可以追加多个元素：x.extend([7,8,(2,3)])

   ​	extend追加元素比加号要快

3. 排序

   ​	原地排序，不创建一个新的对象：a.sort()

   ​	sort中可以指定排序规则：a.sort(key = len)

   ​	排序，产生一个排好序的副本：sorted()

4. 切片

   ​	用切片可以选去大多数序列类型的一部分，切片的基本形式是在方括号中使用start:stop：

   ​	也可以通过切片来对list赋值

   ​	start和stop可以被省略，默认序列的开头和结尾

   ​	负数表示从后向前切片

###### 常用函数

   ​	enumerate函数：

   ​			for i,value in enumerate(collection):

   ​				#do something with value

   ​	sorted函数：

   ​			可以从任意序列的元素返回一个新的排好序的列表

   ​	zip函数:

   ​			将多个列表，元组或其他序列对组合成一个元组列表

   ​	reversed函数:

   ​			从后向前迭代一个序列。reversed是一个生成器，只有实体化(即列表或for循环)之后才能创建反转的序列。

   ​	字典：hashmap

   ​			创建：dl={'a' : 'some value' , 'b' : [1,2,3,4]}

   ​			增加：dl[7]='an integer'

   ​			删除：del关键字，pop(返回值的同时删除键)

   ​			用序列创建字典：
   ```python
   					mapping = {}
    					for key,value in zip(key_list , value_list):
   							mapping[key] = value
   ```
    			默认值：
   ```python
   					if key in some_dict:
   							value = some_dict[key]
   					else:
   							value = default_value
   ```

   ###### 集合set()

   ​		集合是无序的不可重复的元素的集合。set()

   ​				求并集：a.union(b)

   ​				求交集：a.intersection(b)

   ​				还有一堆操作方法，用到的时候查。

   ###### 列表式推导

   ​		允许用户方便的从一个集合过滤元素，形成列表，在传递参数的过程中还可以修改元素。形式如下：[expr for val in collection if condition]

   ​		它相当于：result[] 

   ​							for val in collection:

   ​										if condition:

   ​												result.append(exer)

   ```python
   strings = ['a','as','bat','car','dove','python']
   [x.upper() for x in strings if len(x) > 2]
   ```

###### 集合式推到

和列表式推到特别像，只不过是大括号。

​		set_comp={expr for val in collection if condition}

###### 嵌套列表式推导

用到再查。

#### 3.2函数

```python
#举例
def my_function(x,y,z=1.5):
    if z>1:
        return z*(x+y)
    else:
        return z/(x+y)
```

##### 特殊点

1.可以返回多个值

2.函数也是对象

##### 匿名（lambda）函数

def apply_to_list(some_list, f):

​		return [f(x) for x in some_list]

ints = [4,0,1,5,6]

apply_to_list(ints, lambda x:x*2)

##### 柯里化（currying）:部分参数应用

def add_numbers(x , y):

​		return x+y

add_five = lambda y: add_numbers(5 , y)

##### 生成器

some_dict = {'a' : 1, 'b' : 2, 'c' : 3}

for key in some_dict:

​		print(key)

生成一个迭代器：迭代器是一个特殊的对象，它可以在for循环之类的上下文中向Python解释器输送对象。

dict_iterator  = iter(some_dict)

list(dict_iterator)

###### 生成器表达式

###### itertools 模块

#### 3.3文件系统

略

### 第四章 NumPy基础：数组和矢量计算

#### 4.1NumPy 的ndarray

import numpy as np 

data=np.random.randn(2,3)

data * 10

data + data

data.shape 表示各维度大小

data.dtype 表述数据类型

###### 创建ndarray

np.array(data)

###### 数组的运算

可以进行加、减、乘、除、平方、bool等运算。

###### 基本索引和切片

切片操作得到的是，原数组的一个视图，而不是备份。

```python
arr = array(1,2,3,4,5,6,7,8,9,10)
arr_slice=arr[5:8]
arr_slice[:] = 12
```

当修改arr_slice上的数据时，arr数据数据也会改变。

如果你想得到一个副本，需要使用arr[5:8].copy()

###### 切片索引

arr1d[1:3]

arr2d[:2 , 1: ]

###### 布尔型索引

###### 花式索引

###### 数组转置和轴对换

arr=np.arange(15).reshape((3,5))

求内积np.dot(arr.T , arr)

4.2通用函数

np.sqrt()

np.exp()

np.maximum(x,y)

remainder,whole_part=np.modf() 取整数部分和小数部分

###### 一元ufunc

abs，fabs, sqrt, exp, log(log10、log2、log1p), sign, cell, floor, 

arccos, arccosh, arcsin, arcsinh. arctan , arctanh  

###### 二元ufunc

add ， subtract , multiply , divide , floor_divide , power , maximum , fmax

minimum , fmin , mod , copysign

#### 4.3利用数组处理数据

np.meshgrid()接受两个一维数组，并产生两个二维矩阵

np.where函数是三元表达式，x if condition else y 的矢量化版本。

###### 数学和统计方法

聚合计算：mean() sum() std(),var() min() max() argmin() argmax()分别为最大和最小元素的索引

arr.cumsum(axis=0) 以第一行为基准求和

arr.cumprod(axis=1)以第一列为基准求积

###### 排序

arr.sort()

arr.sort(1)  多维数组在任意一个轴向上排序，只需将轴编号传给sort即可

顶级方法 np.sort() 返回的是数组的已排序副本。就地排序则会修改数据本身。

###### 唯一化以及它的集合逻辑

np.unique(arr) 去除重复项

np.inld(arr1,arr2)判断arr1数组中的值是否在arr2中，返回一个bool数组

#### 4.4用于数组的文件输入输出

```pyth
arr = np.arange(10)
np.save('some_array',arr)
以未压缩的原始二进制格式保存在扩展名为.npy的文件中.npy可以省略
np.load('some_array.npy')
savaz可以将多个数组保存到一个未压缩文件中，将数组以关键字参数的形式传入
np.savaz（'array_archive.npz'，a=arr, b=arr）
如果数据压缩的很好，可以使用np.savaz_compressed
np.savaz_compressed（'array_archive.npz'，a=arr, b=arr）
```

#### 4.5线性代数

x.dot(y)  相当于 np.dat(x , y)

#### 4.6伪随机数生成

seed 确定随机数生成器的种子

permutation 返回一个序列的随机排列或返回一个随机排列的范围

shuffler 对一个序列就地随机排序

rand 产生均匀分布的样本值

randint 从给定的上下限范围内随机选取证书

randn 产生正态分布的样本值

binomial 产生二项分布的样本值

normal 产生正态分布的样本值

beta 产生Beta分布的样本值

chisquare 产生卡方分布的样本值

gamma 产生Gamma分布的样本值

uniform 产生在（0，1)中均匀分布的样本值

