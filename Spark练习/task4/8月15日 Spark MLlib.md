# 8月15日 Spark MLlib

## MLlib

### 介绍

机器学习可以看做是一门人工智能的科学，该领域的主要研究对象是人工智能。机器学习利用数据或以往的经验，以此优化计算机程序的性能标准。一种经常引用的英文定义是：

```
A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E。
```

![](F:\Typora_repo\SparkMLlib\ML.jpg)

机器学习强调三个关键词：算法、经验、性能，其处理过程如上图所示。在数据的基础上，通过算法构建出模型并对模型进行评估。评估的性能如果达到要求，就用该模型来测试其他的数据；如果达不到要求，就要调整算法来重新建立模型，再次进行评估。如此循环往复，最终获得满意的经验来处理其他的数据。机器学习技术和方法已经被成功应用到多个领域，比如个性推荐系统，金融反欺诈，语音识别，自然语言处理和机器翻译，模式识别，智能控制等

### 二、基于大数据的机器学习

在大数据上进行机器学习，需要处理全量数据并进行大量的迭代计算，这要求机器学习平台具备强大的处理能力。Spark 立足于内存计算，天然的适应于迭代式计算。即便如此，对于普通开发者来说，实现一个分布式机器学习算法仍然是一件极具挑战的事情。幸运的是，Spark提供了一个基于海量数据的机器学习库，它提供了常用机器学习算法的分布式实现，开发者只需要有 Spark 基础并且了解机器学习算法的原理，以及方法相关参数的含义，就可以轻松的通过调用相应的 API 来实现基于海量数据的机器学习过程。其次，Spark-Shell的即席查询也是一个关键。算法工程师可以边写代码边运行，边看结果。spark提供的各种高效的工具正使得机器学习过程更加直观便捷。比如通过sample函数，可以非常方便的进行抽样。当然，Spark发展到后面，拥有了实时批计算，批处理，算法库，SQL、流计算等模块等，基本可以看做是全平台的系统。把机器学习作为一个模块加入到Spark中，也是大势所趋。

### 三、Spark机器学习库MLLib

MLLib是Sark的机器学习库，包括分类，回归，聚类，协同过滤，降维等，同时还包括底层的优化原语和高层的管道API。具体来说，其主要包括以下几方面的内容：

1. 算法工具：常用的学习算法，如分类、回归、聚类和协同过滤；
2. 特征化工具：特征提取、转化、降维，和选择工具；
3. 管道(Pipeline)：用于构建、评估和调整机器学习管道的工具;
4. 持久性：保存和加载算法，模型和管道;
5. 实用工具：线性代数，统计，数据处理等工具。

Spark 机器学习库从 1.2 版本以后被分为两个包：

- spark.mllib 包含基于RDD的原始算法API。Spark MLlib 历史比较长，在1.0 以前的版本即已经包含了，提供的算法实现都是基于原始的 RDD。
- spark.ml 则提供了基于DataFrames 高层次的API，可以用来构建机器学习工作流（PipeLine）。ML Pipeline 弥补了原始 MLlib 库的不足，向用户提供了一个基于 DataFrame 的机器学习工作流式 API 套件。

mllib从Spark2.0开始，基于RDD的api开始进入维护模式，并预期于3.0版本移除。

Spark在机器学习方面的发展非常快，目前已经支持了主流的统计和机器学习算法。纵观所有基于分布式架构的开源机器学习库，MLlib可以算是计算效率最高的。MLlib目前支持4种常见的机器学习问题: 分类、回归、聚类和协同过滤。下表列出了目前MLlib支持的主要的机器学习算法：

![](F:\Typora_repo\SparkMLlib\MLTable.png)

## 工作流

### 基本概念

DataFrame:使用Spark SQL中的DataFrame作为数据集，它可以容纳各种数据类型。 较之 RDD，包含了 schema 信息，更类似传统数据库中的二维表格。它被 ML Pipeline 用来存储源数据。例如，DataFrame中的列可以是存储的文本，特征向量，真实标签和预测的标签等。

Transformer：翻译成转换器，是一种可以将一个DataFrame转换为另一个DataFrame的算法。比如一个模型就是一个 Transformer。它可以把 一个不包含预测标签的测试数据集 DataFrame 打上标签，转化成另一个包含预测标签的 DataFrame。技术上，Transformer实现了一个方法transform（），它通过附加一个或多个列将一个DataFrame转换为另一个DataFrame

Estimator：翻译成估计器或评估器，它是学习算法或在训练数据上的训练方法的概念抽象。在 Pipeline 里通常是被用来操作 DataFrame 数据并生产一个 Transformer。从技术上讲，Estimator实现了一个方法fit（），它接受一个DataFrame并产生一个转换器。如一个随机森林算法就是一个 Estimator，它可以调用fit（），通过训练特征数据而得到一个随机森林模型

Parameter：Parameter 被用来设置 Transformer 或者 Estimator 的参数。现在，所有转换器和估计器可共享用于指定参数的公共API。ParamMap是一组（参数，值）对。

PipeLine：翻译为工作流或者管道。工作流将多个工作流阶段（转换器和估计器）连接在一起，形成机器学习的工作流，并获得结果输出。

### 工作流如何工作

要构建一个Pipeline工作流，首先需要定义Pipeline中的各个工作流阶段PipelineStage,(包括转化器和评估器)，比如指标提取和转换模型训练等。有了这些处理特定问题的转换器和评估器，就可以按照具体的处理逻辑有序的组织PipelineStages 并创建一个Pipeline。比如：

 ```
pipeline =Pipeline(stages=[stage1,stage2,stage3])
 ```

然后就可以把训练数据集作为输入参数，调用 Pipeline 实例的 fit 方法来开始以流的方式来处理源训练数据。这个调用会返回一个 PipelineModel 类实例，进而被用来预测测试数据的标签。更具体的说，工作流的各个阶段按顺序运行，输入的DataFrame在它通过每个阶段时被转换。 对于Transformer阶段，在DataFrame上调用transform（）方法。 对于估计器阶段，调用fit（）方法来生成一个转换器（它成为PipelineModel的一部分或拟合的Pipeline），并且在DataFrame上调用该转换器的transform()方法。

![](F:\Typora_repo\SparkMLlib\ml-Pipeline.png)

上面，顶行表示具有三个阶段的流水线。 前两个（Tokenizer和HashingTF）是Transformers（蓝色），第三个（LogisticRegression）是Estimator（红色）。 底行表示流经管线的数据，其中圆柱表示DataFrames。 在原始DataFrame上调用Pipeline.fit（）方法，它具有原始文本文档和标签。 Tokenizer.transform（）方法将原始文本文档拆分为单词，向DataFrame添加一个带有单词的新列。 HashingTF.transform（）方法将字列转换为特征向量，向这些向量添加一个新列到DataFrame。 现在，由于LogisticRegression是一个Estimator，Pipeline首先调用LogisticRegression.fit（）产生一个LogisticRegressionModel。 如果流水线有更多的阶段，则在将DataFrame传递到下一个阶段之前，将在DataFrame上调用LogisticRegressionModel的transform（）方法。

值得注意的是，工作流本身也可以看做是一个估计器。在工作流的fit（）方法运行之后，它产生一个PipelineModel，它是一个Transformer。 这个管道模型将在测试数据的时候使用。 下图说明了这种用法。

![](F:\Typora_repo\SparkMLlib\ml-PipelineModel.png)

在上图中，PipelineModel具有与原始流水线相同的级数，但是原始流水线中的所有估计器都变为变换器。 当在测试数据集上调用PipelineModel的transform（）方法时，数据按顺序通过拟合的工作流。 每个阶段的transform（）方法更新数据集并将其传递到下一个阶段。工作流和工作流模型有助于确保培训和测试数据通过相同的特征处理步骤

### 构建一个机器学习工作流

在jupyter中task4.ipynb中实现

### 特征抽取—TF-IDF

特征提取：从原始数据中抽取特征

特征转换：特征的维度，特征的转化，特征的修改

特征选取：从大规模特征集中选取一个子集。

##### 特征提取

"词频 - 逆向文件频率"(TF-IDF)是一种文本挖掘中广泛使用的特征向量化方法，他可以体现一个文档中词语在语料库的中的重要程度。

 词语由t表示，文档由d表示，语料库由D表示。词频TF(t,d)是词语t在文档d中出现的次数。文件频率DF(t,D)是包含词语的文档的个数。如果我们只使用词频来衡量重要性，很容易过度强调在文档中经常出现，却没有太多实际信息的词语，比如“a”，“the”以及“of”。如果一个词语经常出现在语料库中，意味着它并不能很好的对文档进行区分。TF-IDF就是在数值化文档信息，衡量词语能提供多少信息以区分文档。其定义如下：

 此处 是语料库中总的文档数。公式中使用log函数，当词出现在所有文档中时，它的IDF值变为0。加1是为了避免分母为0的情况。TF-IDF 度量值表示如下：

$IDF(t,D)=log\frac{|D|+1}{DF(t,D)+1}$

此处$D$是语料库中总的文档数。公式中使用log函数，当词出现在所有文档中时，他的IDF值变为0。加1是为了避免分母为0的状态。TF-IDF的度量值表示如下：

$TFIDF(t,d,D)=TF(t,d)*IDF(t,d)$

 在Spark ML库中，TF-IDF被分成两部分：TF (+hashing) 和 IDF。

TF: HashingTF 是一个Transformer，在文本处理中，接收词条的集合然后把这些集合转化成固定长度的特征向量。这个算法在哈希的同时会统计各个词条的词频。

IDF: IDF是一个Estimator，在一个数据集上应用它的fit（）方法，产生一个IDFModel。 该IDFModel 接收特征向量（由HashingTF产生），然后计算每一个词在文档中出现的频次。IDF会减少那些在语料库中出现频率较高的词的权重。

Spark.mllib 中实现词频率统计使用特征hash的方式，原始特征通过hash函数，映射到一个索引值。后面只需要统计这些索引值的频率，就可以知道对应词的频率。这种方式避免设计一个全局1对1的词到索引的映射，这个映射在映射大量语料库时需要花费更长的时间。但需要注意，通过hash的方式可能会映射到同一个值的情况，即不同的原始特征通过Hash映射后是同一个值。为了降低这种情况出现的概率，我们只能对特征向量升维。i.e., 提高hash表的桶数，默认特征维度是 2^20 = 1,048,576.

在下面的代码段中，我们以一组句子开始。首先使用分解器Tokenizer把句子划分为单个词语。对每一个句子（词袋），我们使用HashingTF将句子转换为特征向量，最后使用IDF重新调整特征向量。这种转换通常可以提高使用文本特征的性能。

首先，导入TFIDF所需要的包：

```python
 from pyspark.ml.feature import HashingTF,IDF,Tokenizer
```

准备工作完成后，我们创建一个简单的DataFrame，每一个句子代表一个文档。

```python
sentenceData = spark.createDataFrame([(0, "I heard about Spark and I love Spark"),(0, "I wish Java could use case classes"),(1, "Logistic regression models are neat")]).toDF("label", "sentence")
```

在得到文档集合后，即可用tokenizer对句子进行分词

```python
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")wordsData = tokenizer.transform(sentenceData)
```

得到分词后的文档序列后，即可使用HashingTF的transform()方法把句子哈希成特征向量，这里设置哈希表的桶数为2000。

```python
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)featurizedData = hashingTF.transform(wordsData)
```

可以看到，分词序列被变换成一个稀疏特征向量，其中每个单词都被散列成了一个不同的索引值，特征向量在某一维度上的值即该词汇在文档中出现的次数。

最后，使用IDF来对单纯的词频特征向量进行修正，使其更能体现不同词汇对文本的区别能力，IDF是一个Estimator，调用fit()方法并将词频向量传入，即产生一个IDFModel。

```python
idf = IDF(inputCol="rawFeatures", outputCol="features")idfModel = idf.fit(featurizedData)
```

很显然，IDFModel是一个Transformer，调用它的transform()方法，即可得到每一个单词对应的TF-IDF度量值。

```python
rescaledData = idfModel.transform(featurizedData)rescaledData.select("label", "features").show()
```

可以看到，特征向量已经被其在语料库中出现的总次数进行了修正，通过TF-IDF得到的特征向量，在接下来可以被应用到相关的机器学习方法中。

##### 特征转换



##### 特征选取

