# Spark学习笔记二

### RDD创建

读取外部数据集。（本地文件，HDFS文件系统，HBase,Cassandra,Amazon S3等外部数据源中加载数据集。Spark可以支持文本文件、SequenceFile文件（Hadoop提供的 SequenceFile是一个由二进制序列化过的key/value的字节流组成的文本存储文件）和其他符合Hadoop InputFormat格式的文件）

调用SparkContex的parallelize方法，在Driver中一个已经存在的集合（数组）上创建

##### 从文件系统中加载数据创建RDD

spark采用textFile()方法来从文件系统中加载数据创建RDD。

```python
lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
```

1. 如果使用了本地文件系统的路径，那么，必须要保证在所有的worker节点上，也都能够采用相同的路径访问到该文件，比如，可以把该文件拷贝到每个worker节点上，或者也可以使用网络挂载共享文件系统。

2. textFile()方法的输入参数，可以是文件名，也可以是目录，也可以是压缩文件等。比如，textFile(“/my/directory”), textFile(“/my/directory/*.txt”), and textFile(“/my/directory/*.gz”).

3. textFile()方法也可以接受第2个输入参数（可选），用来指定分区的数目。默认情况下，Spark会为HDFS的每个block创建一个分区（HDFS中每个block默认是128MB）。你也可以提供一个比block数量更大的值作为分区数目，但是，你不能提供一个小于block数量的值作为分区数目。

##### 通过并行集合（数组）创建RDD

可以调用sparkContext的parallelize方法，在Driver中一个已经存在的集合（数组）上创建

```python
nums = [1,2,3,4,5]
rdd = sc.parallelize(nums)
```

上面使用列表来创建。在Python中并没有数组这个基本数据类型，为了便于理解，你可以把列表当成其他语言的数组。

### RDD操作

##### 转换操作

基于现有的数据集创建一个新的数据集

对于RDD而言，每一次转换操作都会产生不同的RDD，供给下一个“转换”使用。转换得到的RDD是惰性求值的，也就是说，整个转换过程只是记录了转换的轨迹，并不会发生真正的计算，只有遇到行动操作时，才会发生真正的计算，开始从血缘关系源头开始，进行物理的转换操作。
下面列出一些常见的转换操作（Transformation API）：

* filter(func)：筛选出满足函数func的元素，并返回一个新的数据集
* map(func)：将每个元素传递到函数func中，并将结果返回为一个新的数据集
* flatMap(func)：与map()相似，但每个输入元素都可以映射到0或多个输出结果
* groupByKey()：应用于(K,V)键值对的数据集时，返回一个新的(K, Iterable)形式的数据集
* reduceByKey(func)：应用于(K,V)键值对的数据集时，返回一个新的(K, V)形式的数据集，其中的每个值是将每个key传递到函数func中进行聚合



##### 行动操作

在数据集上进行运算，返回计算值

行动操作是真正触发计算的地方。Spark程序执行到行动操作时，才会执行真正的计算，从文件中加载数据，完成一次又一次转换操作，最终，完成行动操作得到结果。
下面列出一些常见的行动操作（Action API）：
* count() 返回数据集中的元素个数
* collect() 以数组的形式返回数据集中的所有元素
* first() 返回数据集中的第一个元素
* take(n) 以数组的形式返回数据集中的前n个元素
* reduce(func) 通过函数func（输入两个参数并返回一个值）聚合数据集中的元素
* foreach(func) 将数据集中的每个元素传递到函数func中运行*

### RDD具体案例

#### 两种创建

从文件中加载

```python
>>>  lines = sc.textFile("file:///usr/local/spark/mycode/pairrdd/word.txt")
>>> pairRDD = lines.flatMap(lambda line : line.split(" ")).map(lambda word : (word,1))
```

数组加载

```python
>>> list = ["Hadoop","Spark","Hive","Spark"]
>>> rdd = sc.parallelize(list)

```

#### 键值对转换

##### reduceByKey(func)

##### groupByKey()

##### keys()

##### values()

##### sortBykey()

##### mapValues(func)

##### join()

### 共享变量

##### 广播变量

##### 累加器

### 文件读取

##### 本地文件系统

##### HDFS

