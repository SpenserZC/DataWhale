## 4.1 MySQL 实战
### #学习内容#
数据导入导出
  * 将之前创建的任意一张MySQL表导出,且是CSV格式
  * 再将CSV表导入数据库
### #作业#
项目七: 各部门工资最高的员工（难度：中等）
创建 Employee 表，包含所有员工信息，每个员工有其对应的 Id, salary 和 department Id。
+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
+----+-------+--------+--------------+
创建 Department 表，包含公司所有部门的信息。
+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+
编写一个 SQL 查询，找出每个部门工资最高的员工。例如，根据上述给定的表格，Max 在 IT 部门有最高工资，Henry 在 Sales 部门有最高工资。
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| Sales      | Henry    | 80000  |
+------------+----------+--------+

### 答案
```sql
---建表---
CREATE TABLE Employee(Id INT,
		NAME VARCHAR(10),
		Salary INT,
		DepartmentId INT); 
CREATE TABLE Department(Id INT,
		NAME VARCHAR(10));
---插值---
INSERT INTO Employee VALUES (1,'Joe',70000,1),
			(2,'Henry',80000,2),
			(3,'Sam',60000,2),
			(4,'Max',90000,1);
INSERT INTO Department VALUES(1,'IT'),(2,'Sales');
---查找---
---查找部门最高工资---
SELECT d.Name AS Department,e.name AS Employee,e.Salary FROM 
	Employee e LEFT JOIN department d ON e.DepartmentId=d.Id
	WHERE e.Salary=(SELECT MAX(Salary) FROM Employee WHERE DepartmentId=d.Id);
---查找部门工资最高前三---
SELECT Department.Name AS Department, e1.Name AS Employee, e1.Salary AS Salary
FROM Employee e1
JOIN Department
ON e1.DepartmentId = Department.Id
WHERE 3 > (
SELECT COUNT(DISTINCT e2.Salary) 
FROM Employee e2
WHERE e2.Salary > e1.Salary AND e1.DepartmentId = e2.DepartmentId
)
ORDER BY Department.Name, e1.Salary DESC
```


项目八: 换座位（难度：中等）
小美是一所中学的信息科技老师，她有一张 seat 座位表，平时用来储存学生名字和与他们相对应的座位 id。
其中纵列的 **id **是连续递增的
小美想改变相邻俩学生的座位。
你能不能帮她写一个 SQL query 来输出小美想要的结果呢？
 请创建如下所示 seat 表：
**示例：**
+---------+---------+
|    id   | student |
+---------+---------+
|    1    | Abbot   |
|    2    | Doris   |
|    3    | Emerson |
|    4    | Green   |
|    5    | Jeames  |
+---------+---------+
假如数据输入的是上表，则输出结果如下：
+---------+---------+
|    id   | student |
+---------+---------+
|    1    | Doris   |
|    2    | Abbot   |
|    3    | Green   |
|    4    | Emerson |
|    5    | Jeames  |
+---------+---------+
**注意：**
如果学生人数是奇数，则不需要改变最后一个同学的座位。
```sql
---建表插值---
CREATE TABLE seat(id INT,Student VARCHAR(10));
INSERT INTO seat VALUES(1,'Abbot'),
			(2,'Doris'),
			(3,'Emerson'),
			(4,'Green'),
			(5,'Jeames');
---转换座位---
SELECT(	CASE
	WHEN id %2 = 1 AND id!=max_id THEN id+1
	WHEN id %2 = 0 THEN id-1
	WHEN id = max_id THEN id
	END) AS id,student 
	FROM(SELECT id, student, (SELECT MAX(id) FROM seat) AS max_id FROM seat) a
	ORDER BY id;
```

项目九: 分数排名（难度：中等）
编写一个 SQL 查询来实现分数排名。如果两个分数相同，则两个分数排名（Rank）相同。请注意，平分后的下一个名次应该是下一个连续的整数值。换句话说，名次之间不应该有“间隔”。
创建以下 score 表：
+----+-------+
| Id | Score |
+----+-------+
| 1 | 3.50 |
| 2 | 3.65 |
| 3 | 4.00 |
| 4 | 3.85 |
| 5 | 4.00 |
| 6 | 3.65 |
+----+-------+
例如，根据上述给定的 scores 表，你的查询应该返回（按分数从高到低排列）：
+-------+------+
| Score | Rank |
+-------+------+
| 4.00 | 1 |
| 4.00 | 1 |
| 3.85 | 2 |
| 3.65 | 3 |
| 3.65 | 3 |
| 3.50 | 4 |
+-------+------+
### 答案
```sql
---建表，插值---
CREATE TABLE Scores (Id INT,Score DOUBLE);
INSERT INTO Scores VALUES(1,3.50),(2,3.65),(3,4.00),
			(4,3.85),(5,4.00),(6,3.65);
----查询--
SELECT u.score,a.rank AS Rank FROM Scores u,
(SELECT @counter:=@counter+1 AS rank,t.score FROM (SELECT @counter:=0,score FROM Scores GROUP BY score ORDER BY score DESC)AS t)a
WHERE u.score=a.score ORDER BY Rank ASC;
```

## 4.2 MySQL 实战 - 复杂项目
### #作业#
项目十：行程和用户（难度：困难）
Trips 表中存所有出租车的行程信息。每段行程有唯一键 Id，Client_Id 和 Driver_Id 是 Users 表中 Users_Id 的外键。Status 是枚举类型，枚举成员为 (‘completed’, ‘cancelled_by_driver’, ‘cancelled_by_client’)。
+----+-----------+-----------+---------+--------------------+----------+
| Id | Client_Id | Driver_Id | City_Id |        Status      |Request_at|
+----+-----------+-----------+---------+--------------------+----------+
| 1  |     1     |    10     |    1    |     completed      |2013-10-01|
| 2  |     2     |    11     |    1    | cancelled_by_driver|2013-10-01|
| 3  |     3     |    12     |    6    |     completed      |2013-10-01|
| 4  |     4     |    13     |    6    | cancelled_by_client|2013-10-01|
| 5  |     1     |    10     |    1    |     completed      |2013-10-02|
| 6  |     2     |    11     |    6    |     completed      |2013-10-02|
| 7  |     3     |    12     |    6    |     completed      |2013-10-02|
| 8  |     2     |    12     |    12   |     completed      |2013-10-03|
| 9  |     3     |    10     |    12   |     completed      |2013-10-03| 
| 10 |     4     |    13     |    12   | cancelled_by_driver|2013-10-03|
+----+-----------+-----------+---------+--------------------+----------+
Users 表存所有用户。每个用户有唯一键 Users_Id。Banned 表示这个用户是否被禁止，Role 则是一个表示（‘client’, ‘driver’, ‘partner’）的枚举类型。
+----------+--------+--------+
| Users_Id | Banned |  Role  |
+----------+--------+--------+
|    1     |   No   | client |
|    2     |   Yes  | client |
|    3     |   No   | client |
|    4     |   No   | client |
|    10    |   No   | driver |
|    11    |   No   | driver |
|    12    |   No   | driver |
|    13    |   No   | driver |
+----------+--------+--------+
写一段 SQL 语句查出 **2013年10月1日 **至 **2013年10月3日 **期间非禁止用户的取消率。基于上表，你的 SQL 语句应返回如下结果，取消率（Cancellation Rate）保留两位小数。
+------------+-------------------+
|     Day    | Cancellation Rate |
+------------+-------------------+
| 2013-10-01 |       0.33        |
| 2013-10-02 |       0.00        |
| 2013-10-03 |       0.50        |
+------------+-------------------+

### 答案
```sql
---创建插值---
CREATE TABLE Trips(Id INT,Client_Id INT,Driver_id INT,City_Id INT,STATUS VARCHAR(10),Request_at DATETIME);

INSERT INTO Trips VALUES(1,1,10,1 ,'completed','2013-10-01'),
		(2,2,11,1 ,'cancelled_by_driver','2013-10-01'),
		(3,3,12,6 ,'completed','2013-10-01'),
		(4,4,13,6 ,'cancelled_by_client','2013-10-01'),
		(5,1,10,1 ,'completed','2013-10-02'),
		(6,2,11,6 ,'completed','2013-10-02'),
		(7,3,12,6 ,'completed','2013-10-02'),
		(8,2,12,12,'completed','2013-10-03'),
		(9,3,10,12,'completed','2013-10-03'),
		(1,4,13,12,'cancelled_by_driver','2013-10-03');

CREATE TABLE Users(Users_Id INT,Banned VARCHAR(3),Role VARCHAR(6));

INSERT INTO Users VALUES (1 ,'No','client'),
		(2 ,'Yes','client'),
		(3 ,'No','client'),
		(4 ,'No','client'),
		(10,'No','driver'),
		(11,'No','driver'),
		(12,'No','driver'),
		(13,'No','driver');

---查询---
SELECT T2.DAY,IFNULL(ROUND((T1.num/T2.num),2),0) AS 'Cancellation Rate'
FROM
(SELECT Request_at as Day,count(*) as num
	FROM Trips t
	LEFT JOIN Users u
	ON t.Client_Id = u.Users_Id
  WHERE u.Banned != 'Yes'
  AND t.status != 'completed'
  AND Request_at >='2013-10-01' AND Request_at <= '2013-10-03'
	GROUP BY Day) AS T1
RIGHT JOIN
 (SELECT Request_at as Day,count(*) as num
	FROM Trips t
	LEFT JOIN Users u
	ON t.Client_Id = u.Users_Id
  WHERE u.Banned != 'Yes'
  AND Request_at >='2013-10-01' AND Request_at <= '2013-10-03'
	GROUP BY Day) AS T2
  ON T1.DAY = T2.DAY;
```



项目十一：各部门前3高工资的员工（难度：中等）
将项目7中的 employee 表清空，重新插入以下数据（其实是多插入5,6两行）：
+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
| 5  | Janet | 69000  | 1            |
| 6  | Randy | 85000  | 1            |
+----+-------+--------+--------------+
编写一个 SQL 查询，找出每个部门工资前三高的员工。例如，根据上述给定的表格，查询结果应返回：
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| IT         | Randy    | 85000  |
| IT         | Joe      | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |
+------------+----------+--------+
此外，请考虑实现各部门前N高工资的员工功能。

```sql
---查找前三---
SELECT Department.Name AS Department, e1.Name AS Employee, e1.Salary AS Salary
FROM Employee e1
JOIN Department
ON e1.DepartmentId = Department.Id
WHERE 3 > (
SELECT COUNT(DISTINCT e2.Salary) 
FROM Employee e2
WHERE e2.Salary > e1.Salary AND e1.DepartmentId = e2.DepartmentId
)
ORDER BY Department.Name, e1.Salary DESC

---查找前N的排名，把where后3改成N就好---
```



项目十二  分数排名 - （难度：中等）
依然是昨天的分数表，实现排名功能，但是排名是非连续的，如下：
+-------+------+
| Score | Rank |
+-------+------+
| 4.00  | 1    |
| 4.00  | 1    |
| 3.85  | 3    |
| 3.65  | 4    |
| 3.65  | 4    |
| 3.50  | 6    |
+-------+------
```sql
---查询---
SELECT s.score,(SELECT COUNT(*)+1 FROM scores AS s1 WHERE 					s1.score>s.score) AS rank
	FROM scores s ORDER BY score DESC;
```
