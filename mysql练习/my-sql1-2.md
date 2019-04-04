# SQL练习（一）
说明：最近面试，深感sql方面需要强化训练。恰巧，在机缘巧合的情况下（忘记是怎么加入的），一个mysql训练营，每天会发布一定的任务，看起来还不错的样子。那，就一起参加一波呗。
## 任务与解答
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

编写一个 SQL 查询，查找 email 表中所有重复的电子邮箱。
根据以上输入，你的查询应返回以下结果：
```
+---------+
| Email   |
+---------+
| a@b.com |
+---------+
```
**说明：**所有电子邮箱都是小写字母。
### 答案：
```sql
建表:
CREATE TABLE email(id INT PRIMARY KEY,Email VARCHAR(20));
插入数据：
INSERT INTO email(id,Email) VALUES(1,'a@b.com'),(2,'c@d.com'),(3,'a@b.com'); 
查找：重复的电子邮箱
SELECT Email FROM email GROUP BY Email HAVING COUNT(id)>1;
```


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
### 解答
```sql
建表：
CREATE TABLE world(NAME VARCHAR(20),continent VARCHAR(20), AREA INT ,population INT,gdp INT );
插入数据：
INSERT INTO World
  VALUES('Afghanistan','Asia',652230,25500100,20343000),
		('Albania','Europe',28748,2831741,12960000),
		('Algeria','Africa',2381741,37100000,188681000),
		('Andorra','Europe',468,78115,3712000),
		('Angola','Africa',1246700,20609294,100990000);
查询：
SELECT NAME,population,AREA FROM world WHERE AREA>3000000 OR (population>25000000 AND gdp>20000000);
```




