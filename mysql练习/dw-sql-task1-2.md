负责人：江水
# Task 1(3天)
## 1.1 - MySQL 软件安装及数据库基础
学习内容：
1. 软件安装及服务器设置。[http://www.runoob.com/mysql/mysql-install.html](http://www.runoob.com/mysql/mysql-install.html)
2. 使用图形界面软件 Navicat for SQL
3. 数据库基础知识
  1. 数据库定义
  2. 关系型数据库
  3. 二维表
  4. 行
  5. 列
  6. 主键
  7. 外键
4. MySQL数据库管理系统
  1. 数据库
  2. 数据表
  3. 视图
  4. 存储过程
* 参考资料
  * [SQL必知必会] [https://u18036366.pipipan.com/fs/18036366-300877816](https://u18036366.pipipan.com/fs/18036366-300877816)
  * [MySQL教程] [http://www.runoob.com/mysql/mysql-tutorial.html](http://www.runoob.com/mysql/mysql-tutorial.html)

额外的参考资料：
  * 虚拟机安装Linux [https://blog.csdn.net/yang5726685/article/details/78635388](https://blog.csdn.net/yang5726685/article/details/78635388)
  * Windows 10下 MySQL [https://cloud.tencent.com/developer/article/1010608](https://cloud.tencent.com/developer/article/1010608)
  * Windows 安装 MySQL 常见问题 [https://blog.csdn.net/qq_40942329/article/details/79125366](https://blog.csdn.net/qq_40942329/article/details/79125366)
## 1.2 - MySQL 基础 （一）查询语句
* 导入示例数据库，教程 [https://www.yiibai.com/mysql/how-to-load-sample-database-into-mysql-database-server.html](https://www.yiibai.com/mysql/how-to-load-sample-database-into-mysql-database-server.html)
* SQL是什么？MySQL是什么？

查询语句 SELECT FROM 
  * 语句解释
  * 去重语句
  * 前N个语句
* 筛选语句 WHERE 
  * 语句解释
  * 运算符
* 分组语句 GROUP BY
  * 语句解释
  * HAVING子句
* 排序语句 ORDER BY 
  * 语句解释
  * 正序、逆序
* SQL注释
* SQL代码规范
  * [SQL编程格式的优化建议] [https://zhuanlan.zhihu.com/p/27466166](https://zhuanlan.zhihu.com/p/27466166)
  * [SQL Style Guide][https://www.sqlstyle.guide/](https://www.sqlstyle.guide/)
## 项目一：查找重复的电子邮箱（难度：简单）
创建 email表，并插入如下三行数据
```

+----+---------+
| Id | Email   |
+----+---------+
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
+----+---------+
```

编写一个 SQL 查询，查找 Person 表中所有重复的电子邮箱。
根据以上输入，你的查询应返回以下结果：
```

+---------+
| Email   |
+---------+
| a@b.com |
+---------+
```
**说明：**所有电子邮箱都是小写字母。

## 项目二：查找大国（难度：简单）
创建如下 World 表
```
+------------+----------+---------+--------------+---------------+
| name       | continent| area    | population   | gdp           |
+------------+----------+---------+--------------+---------------+
| Afghanistan| Asia     | 652230  | 25500100     | 20343000      |
| Albania    | Europe   | 28748   | 2831741      | 12960000      |
| Algeria    | Africa   | 2381741 | 37100000     | 188681000     |
| Andorra    | Europe   | 468     | 78115        | 3712000       |
| Angola     | Africa   | 1246700 | 20609294     | 100990000     |
+------------+----------+---------+--------------+---------------+
```
如果一个国家的面积超过300万平方公里，或者(人口超过2500万并且gdp超过2000万)，那么这个国家就是大国家。
编写一个SQL查询，输出表中所有大国家的名称、人口和面积。
例如，根据上表，我们应该输出:
```
+--------------+-------------+--------------+
| name         | population  | area         |
+--------------+-------------+--------------+
| Afghanistan  | 25500100    | 652230       |
| Algeria      | 37100000    | 2381741      |
+--------------+-------------+--------------+
```
# 【任务说明】
1.1是软件安装和配置，以及一些数据库理论知识储备。
学习内容是指需要在博客文章中总结的知识点，包括但不仅限于这些知识点。比如一些安装过程中的报错及解决办法也可以写。

1.2是最最基础的查询语句，可以说学完其中内容，SQL语句就掌握了30%了。
语言规范非常重要，请大家认真仔细阅读。请记住，你写SQL需要考虑别人review时的心情。写的过于杂乱会分分钟造成暴力事件。
学习内容中函数部分，是让大家了解下MySQL可以怎样处理一些数据。了解些常用的，等实际中遇到了再回头查找详细就行。
祝大家学习开心。:-)

考虑到本次集训有很多新手，本次作业赠送建表代码，意不意外，开不开心。
直接将以下code内容复制到cmd或者navicat运行就行。

**项目一**
-- 创建表
```
CREATE TABLE email (
ID INT NOT NULL PRIMARY KEY,
Email VARCHAR(255)
)
```
-- 插入数据
```
INSERT INTO email VALUES('1','a@b.com');
INSERT INTO email VALUES('2','c@d.com');
INSERT INTO email VALUES('3','a@b.com');
```

**项目二**
-- 创建表
```
CREATE TABLE World (
name VARCHAR(50) NOT NULL,
continent VARCHAR(50) NOT NULL,
area INT NOT NULL,
population INT NOT NULL,
gdp INT NOT NULL
);
```
-- 插入数据
```
INSERT INTO World
  VALUES('Afghanistan','Asia',652230,25500100,20343000);
INSERT INTO World 
  VALUES('Albania','Europe',28748,2831741,12960000);
INSERT INTO World 
  VALUES('Algeria','Africa',2381741,37100000,188681000);
INSERT INTO World
  VALUES('Andorra','Europe',468,78115,3712000);
INSERT INTO World
  VALUES('Angola','Africa',1246700,20609294,100990000);
```
# 【打卡说明】
1）**任务一****4.2晚****22:00** 前在下方表格打卡
2）未按时打卡者将会被清退，打卡链接包含：博客/Github链接（描述：任务、遇到的问题、实现代码和参考资料）

⭐注意：【第一个打卡的学员】需要在群内【艾特负责人】，会获得"飞毛腿"奖～
# 【打卡表格】
| **学号**   | **打卡链接**   | 
|:----:|:----|
| **参考**  **(000号)**   | 正常打卡格式：000-昵称-链接   | 
|    | 想自荐优秀作业的打卡格式：自荐-000-昵称-链接   | 
| **001**   |    | 
| **002**   |    | 
| **003**   |    | 
| **004**   |    | 
| **005**   |    | 
| **006**   |    | 
| **007**   |    | 
| **008**   |    | 
| **009**   |    | 
| **010**   |    | 
| **011**   |    | 
| **012**   | [012-A Low B](https://blog.csdn.net/ly294687451/article/details/88928736)[-https://blog.csn.net/ly294687451/article/details/88928736](https://blog.csdn.net/ly294687451/article/details/88928736)   | 
| **013**   |    | 
| **014**   |    | 
| **015**   |    | 
| **016**   |    | 
| **017**   |    | 
| **018**   |    | 
| **019**   |    | 
| **020**   |    | 
| **021**   | 021-TT-[https://blog.csdn.net/weixin_43904850/article/details/88921186](https://blog.csdn.net/weixin_43904850/article/details/88921186)    | 
| **022**   |    | 
| **023**   |    | 
| **024**   |    | 
| **025**   |    | 
| **026**   |    | 
| **027**   |    | 
| **028**   |    | 
| **029**   |    | 
| **030**   |    | 
| **031**   |    | 
| **032**   |    | 
| **033**   |    | 
| **034**   |    | 
| **035**   |    | 
| **036**   |    | 
| **037**   |    | 
| **038**   |    | 
| **039**   |    | 
| **040**   |    | 
| **041**   | 041-Victor-[https://blog.csdn.net/PohhetS2/article/details/88931986](https://blog.csdn.net/PohhetS2/article/details/88931986)   | 
| **042**   |    | 
| **043**   |    | 
| **044**   |    | 
| **045**   |    | 
| **046**   |    | 
| **047**   | 047-aliton-[https://blog.csdn.net/yanghuangsanguo/article/details/88932423](https://blog.csdn.net/yanghuangsanguo/article/details/88932423)   | 
| **048**   |    | 
| **049**   |    | 
| **050**   | 050-回响-[https://www.cnblogs.com/zhgmen/p/10630657.html](https://www.cnblogs.com/zhgmen/p/10630657.html)  第二个完成   | 
| **051**   |    | 
| **052**   | 052-兔斯基-[https://www.jianshu.com/p/c32f3a81e9f0](https://www.jianshu.com/p/c32f3a81e9f0)  第三个完成   | 
| **053**   | 053-T糖[https://blog.csdn.net/weixin_42796244/article/details/88932070](https://blog.csdn.net/weixin_42796244/article/details/88932070)第4个完     | 
| **054**   |    | 
| **055**   |    | 
| **056**   | 056-Jesse：[https://mp.weixin.qq.com/s/VW9Qu8q67F-T_o4aFr2sEA](https://mp.weixin.qq.com/s/VW9Qu8q67F-T_o4aFr2sEA)   | 
| **057**   |    | 
| **058**   |    | 
| **059**   |    | 
| **060**   |    | 
| **061**   |    | 
| **062**   |    | 
| **063**   |    | 
| **064**   |    | 
| **065**   |    | 
| **066**   |    | 
| **067**   |    | 
| **068**   |    | 
| **069**   | 069-GeXeLr-[https://blog.csdn.net/haosuo1486/article/details/88930471](https://blog.csdn.net/haosuo1486/article/details/88930471)   | 
| **070**   |    | 
| **071**   | 071 + Volcano +[https://blog.csdn.net/volcano1995/article/details/88929260](https://blog.csdn.net/volcano1995/article/details/88929260)   | 
| **072**   |    | 
| **073**   |    | 
| **074**   |    | 
| **075**   |    | 
| **076**   |    | 
| **077**   |    | 


