# 8月13日sparksql

### sparksql简介

Shark即Hive on Spark，为了实现与Hive兼容，Shark在HiveQL方面重用了Hive中的HiveQL解析、逻辑执行计划翻译、执行计划优化等逻辑，可以近似认为仅将物理执行计划从MapReduce作业替换成了Spark作业，通过Hive的HiveQL解析，把HiveQL翻译成Spark上的RDD操作。

##### spark sql底层设计

Spark SQL的架构如图16-12所示，在Shark原有的架构上重写了逻辑执行计划的优化部分，解决了Shark存在的问题。Spark SQL在Hive兼容层面仅依赖HiveQL解析和Hive元数据，也就是说，从HQL被解析成抽象语法树（AST）起，就全部由Spark SQL接管了。Spark SQL执行计划生成和优化都由Catalyst（函数式关系查询优化框架）负责。

![](E:\Typora_Config\spark\sparksql\图16-12-Spark-SQL架构.jpg)

​																			图16-12-Spark-SQL架构
​		Spark SQL增加了SchemaRDD（即带有Schema信息的RDD），使用户可以在Spark SQL中执行SQL语句，数据既可以来自RDD，也可以来自Hive、HDFS、Cassandra等外部数据源，还可以是JSON格式的数据。Spark SQL目前支持Scala、Java、Python三种语言，支持SQL-92规范。从Spark1.2 升级到Spark1.3以后，Spark SQL中的SchemaRDD变为了DataFrame，DataFrame相对于SchemaRDD有了较大改变,同时提供了更多好用且方便的API，如图16-13所示。

![](E:\Typora_Config\spark\sparksql\图16-13-Spark-SQL支持的数据格式和编程语言.jpg)

图16-13-Spark-SQL支持的数据格式和编程语言

Spark SQL可以很好地支持SQL查询，一方面，可以编写Spark应用程序使用SQL语句进行数据查询，另一方面，也可以使用标准的数据库连接器（比如JDBC或ODBC）连接Spark进行SQL查询，这样，一些市场上现有的商业智能工具（比如Tableau）就可以很好地和Spark SQL组合起来使用，从而使得这些外部工具借助于Spark SQL也能获得大规模数据的处理分析能力。

##### DataFrame与RDD的区别

DataFrame的推出，让Spark具备了处理大规模结构化数据的能力，不仅比原有的RDD转化方式更加简单易用，而且获得了更高的计算性能。Spark能够轻松实现从MySQL到DataFrame的转化，并且支持SQL查询。

![](E:\Typora_Config\spark\sparksql\DataFrame-RDD.jpg)

图 DataFrame与RDD的区别

从上面的图中可以看出DataFrame和RDD的区别。RDD是分布式的 Java对象的集合，比如，RDD[Person]是以Person为类型参数，但是，Person类的内部结构对于RDD而言却是不可知的。DataFrame是一种以RDD为基础的分布式数据集，也就是分布式的Row对象的集合（每个Row对象代表一行记录），提供了详细的结构信息，也就是我们经常说的模式（schema），Spark SQL可以清楚地知道该数据集中包含哪些列、每列的名称和类型。
		和RDD一样，DataFrame的各种变换操作也采用惰性机制，只是记录了各种转换的逻辑转换路线图（是一个DAG图），不会发生真正的计算，这个DAG图相当于一个逻辑查询计划，最终，会被翻译成物理查询计划，生成RDD DAG，按照之前介绍的RDD DAG的执行方式去完成最终的计算得到结果。

##### DataFrame的创建

从Spark2.0以上版本开始，Spark使用全新的SparkSession接口替代Spark1.6中的SQLContext及HiveContext接口来实现其对数据加载、转换、处理等功能。SparkSession实现了SQLContext及HiveContext所有功能。

SparkSession支持从不同的数据源加载数据，并把数据转换成DataFrame，并且支持把DataFrame转换成SQLContext自身中的表，然后使用SQL语句来操作数据。SparkSession亦提供了HiveQL以及其他依赖于Hive的功能的支持。

##### 从RDD转换的到DataFrame

在jupyter notebook中练习

### 流计算

数据总体上可以分为静态数据和流数据。对静态数据和流数据的处理，对应着两种截然不同的计算模式：批量计算和实时计算。批量计算以“静态数据”为对象，可以在很充裕的时间内对海量数据进行批量处理，计算得到有价值的信息。Hadoop就是典型的批处理模型，由HDFS和HBase存放大量的静态数据，由MapReduce负责对海量数据执行批量计算。流数据必须采用实时计算，实时计算最重要的一个需求是能够实时得到计算结果，一般要求响应时间为秒级。当只需要处理少量数据时，实时计算并不是问题；但是，在大数据时代，不仅数据格式复杂、来源众多，而且数据量巨大，这就对实时计算提出了很大的挑战。因此，针对流数据的实时计算——流计算，应运而生。

总的来说，流计算秉承一个基本理念，即数据的价值随着时间的流逝而降低。因此，当事件出现时就应该立即进行处理，而不是缓存起来进行批量处理。为了及时处理流数据，就需要一个低延迟、可扩展、高可靠的处理引擎。对于一个流计算系统来说，它应达到如下需求。
\* • 高性能。处理大数据的基本要求，如每秒处理几十万条数据。
\* • 海量式。支持TB级甚至是PB级的数据规模。
\* • 实时性。必须保证一个较低的延迟时间，达到秒级别，甚至是毫秒级别。
\* • 分布式。支持大数据的基本架构，必须能够平滑扩展。
\* • 易用性。能够快速进行开发和部署。
\* • 可靠性。能可靠地处理流数据。

##### 流计算框架

Twitter Storm ，Yahoo! S4 ，等等

###### 数据采集服务

数据实时采集阶段通常采集多个数据源的海量数据，需要保证实时性、低延迟与稳定可靠。以日志数据为例，由于分布式集群的广泛应用，数据分散存储在不同的机器上，因此需要实时汇总来自不同机器上的日志数据。目前有许多互联网公司发布的开源分布式日志采集系统均可满足每秒数百MB的数据采集和传输需求，如Facebook的Scribe、LinkedIn的Kafka、淘宝的TimeTunnel，以及基于Hadoop的Chukwa和Flume等。

###### 数据实时计算

流处理系统接收数据采集系统不断发来的实时数据，实时地进行分析计算，并反馈实时结果。

###### 实时查询服务

流计算的第三个阶段是实时查询服务，经由流计算框架得出的结果可供用户进行实时查询、展示或储存。

##### Spark Streaming

Spark Streaming是构建在Spark上的实时计算框架，它扩展了Spark处理大规模流式数据的能力。Spark Streaming可结合批处理和交互查询，适合一些需要对历史数据和实时数据进行结合分析的应用场景。

###### 设计

可整合多种输入源，如Kafka、Flume、HDFS，甚至是普通的TCP套接字。经处理后的数据可存储至文件系统、数据库，或显示在仪表盘里。

基本原理时将实时输入数据流以时间片(秒级)为单位进行拆分，然后经Spark引擎以类似批处理的方式处理每个时间片数据，执行流程如下图所示：

![](E:\Typora_Config\spark\sparksql\图10-20-Spark-Streaming执行流程.jpg)

Spark Streaming 主要的抽象DStream(Discretized Stream,离散化数据流)，表示连续不断的数据流。在内部实现上，Spark Streaming 的输入数据按照时间片（如1S）分成一段段的DStream,灭一段数据转换成为Spark中的RDD，并且对DStream的操作都最终转变为相应的RDD的操作。例如，下图展示了进行单词统计时，每个时间片的数据（存储句子的RDD）经flatMap操作，生成了存储单词的RDD。整个流式计算可根据业务的需求对这些中间的结果进一步处理，或者存储到外部设备中。

![](E:\Typora_Config\spark\sparksql\图10-21-DStream操作示意图.jpg)

###### Spark Streaming与Storm的对比

Spark Streaming和Storm最大的区别在于，Spark Streaming无法实现毫秒级的流计算，而Storm可以实现毫秒级响应。
Spark Streaming无法实现毫秒级的流计算，是因为其将流数据按批处理窗口大小（通常在0.5~2秒之间）分解为一系列批处理作业，在这个过程中，会产生多个Spark 作业，且每一段数据的处理都会经过Spark DAG图分解、任务调度过程，因此，无法实现毫秒级相应。Spark Streaming难以满足对实时性要求非常高（如高频实时交易）的场景，但足以胜任其他流式准实时计算场景。相比之下，Storm处理的单位为Tuple，只需要极小的延迟。
Spark Streaming构建在Spark上，一方面是因为Spark的低延迟执行引擎（100毫秒左右）可以用于实时计算，另一方面，相比于Storm，RDD数据集更容易做高效的容错处理。此外，Spark Streaming采用的小批量处理的方式使得它可以同时兼容批量和实时数据处理的逻辑和算法，因此，方便了一些需要历史数据和实时数据联合分析的特定应用场合。

### 问题：

##### 简述spark sql的运行机制

spark sql运行的核心Catalyst，是一套针对spark sql 语句执行过程的查询优化框架。

SQL语句首先通过Parser模块被解析为语法树，此棵树成为Unresolved Logical Plan; Unresolved Logical Plan 通过Analyzer模块借助于数据元数据解析为Logical Plan；此时再通过基于规则的优化策略进行深入优化，得到Optimized Logical Plan；优化后的逻辑执行计划依然是逻辑的，并不能被Spark系统理解，此时需要将此逻辑执行计划转换为Physical Plan；为了更好的对整个过程进行理解，下文通过一个简单示例进行解释。

###### **Parser**

Parser简单来说是将SQL字符串切分成一个一个Token，再根据一定语义规则解析为一棵语法树。Parser模块目前基本都使用第三方类库ANTLR进行实现，比如Hive、 Presto、SparkSQL等。下图是一个示例性的SQL语句（有两张表，其中people表主要存储用户基本信息，score表存储用户的各种成绩），通过Parser解析后的AST语法树如右图所示：

![](E:\Typora_Config\spark\sparksql\sqlparser\9129747-e5861ad2c1c352b6.png)

###### **Analyzer**

通过解析后的逻辑执行计划基本有了骨架，但是系统并不知道score、sum这些都是些什么鬼，此时需要基本的元数据信息来表达这些词素，最重要的元数据信息主要包括两部分：表的Scheme和基本函数信息，表的scheme主要包括表的基本定义（列名、数据类型）、表的数据格式（Json、Text）、表的物理位置等，基本函数信息主要指类信息。

Analyzer会再次遍历整个语法树，对树上的每个节点进行数据类型绑定以及函数绑定，比如people词素会根据元数据表信息解析为包含age、id以及name三列的表，people.age会被解析为数据类型为int的变量，sum会被解析为特定的聚合函数，如下图所示：

SparkSQL中Analyzer定义了各种解析规则，有兴趣深入了解的童鞋可以查看Analyzer类，其中定义了基本的解析规则，如下：

![](E:\Typora_Config\spark\sparksql\sqlparser\9129747-58987d44bbdf0c02.png)

###### **Optimizer**

![](E:\Typora_Config\spark\sparksql\sqlparser\9129747-3cb65da900d749ef.png)

优化器是整个Catalyst的核心，上文提到优化器分为基于规则优化和基于代价优化两种，当前SparkSQL 2.1依然没有很好的支持基于代价优化（下文细讲），此处只介绍基于规则的优化策略，基于规则的优化策略实际上就是对语法树进行一次遍历，模式匹配能够满足特定规则的节点，再进行相应的等价转换。因此，基于规则优化说到底就是一棵树等价地转换为另一棵树。SQL中经典的优化规则有很多，下文结合示例介绍三种比较常见的规则：谓词下推（Predicate Pushdown）、常量累加（Constant Folding）和列值裁剪（Column Pruning）。

![](E:\Typora_Config\spark\sparksql\sqlparser\9129747-cf227fe1be40e532.png)

上图左边是经过Analyzer解析后的语法树，语法树中两个表先做join，之后再使用age>10对结果进行过滤。大家知道join算子通常是一个非常耗时的算子，耗时多少一般取决于参与join的两个表的大小，如果能够减少参与join两表的大小，就可以大大降低join算子所需时间。谓词下推就是这样一种功能，它会将过滤操作下推到join之前进行，上图中过滤条件age>0以及id!=null两个条件就分别下推到了join之前。这样，系统在扫描数据的时候就对数据进行了过滤，参与join的数据量将会得到显著的减少，join耗时必然也会降低。

![](E:\Typora_Config\spark\sparksql\sqlparser\9129747-04cf4b26df3cab0e.png)

常量累加其实很简单，就是上文中提到的规则  x+(1+2)  -> x+3，虽然是一个很小的改动，但是意义巨大。示例如果没有进行优化的话，每一条结果都需要执行一次100+80的操作，然后再与变量math_score以及english_score相加，而优化后就不需要再执行100+80操作。

列值裁剪是另一个经典的规则，示例中对于people表来说，并不需要扫描它的所有列值，而只需要列值id，所以在扫描people之后需要将其他列进行裁剪，只留下列id。这个优化一方面大幅度减少了网络、内存数据量消耗，另一方面对于列存数据库（Parquet）来说大大提高了扫描效率

![](E:\Typora_Config\spark\sparksql\sqlparser\9129747-b1ef19c961b4c270.png)

除此之外，Catalyst还定义了很多其他优化规则，有兴趣深入了解的童鞋可以查看Optimizer类，下图简单的截取一部分规则：

![](E:\Typora_Config\spark\sparksql\sqlparser\9129747-781924b6b8fd127a.png)

至此，逻辑执行计划已经得到了比较完善的优化，然而，逻辑执行计划依然没办法真正执行，他们只是逻辑上可行，实际上Spark并不知道如何去执行这个东西。比如Join只是一个抽象概念，代表两个表根据相同的id进行合并，然而具体怎么实现这个合并，逻辑执行计划并没有说明。

此时就需要将逻辑执行计划转换为物理执行计划，将逻辑上可行的执行计划变为Spark可以真正执行的计划。比如Join算子，Spark根据不同场景为该算子制定了不同的算法策略，有BroadcastHashJoin、ShuffleHashJoin以及SortMergeJoin等（可以将Join理解为一个接口，BroadcastHashJoin是其中一个具体实现），物理执行计划实际上就是在这些具体实现中挑选一个耗时最小的算法实现，这个过程涉及到基于代价优化策略，后续文章细讲。

参考链接：https://www.jianshu.com/p/efea07f43e7c