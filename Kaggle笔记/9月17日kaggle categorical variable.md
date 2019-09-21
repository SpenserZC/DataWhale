# 9月17日kaggle categorical variable

### categorical variable

明确的变量有确定数量的值

例如吃早餐的频率：从不，有时，大多数，每天

例如拥有车的型号：本田，丰田，福田..

这些变量的值都是确定的，但是不能直接将他们应用到python的模型中。否则大多数的模型都会出错。所以需要预处理；

### Three Approaches

1.Drop categorical variable

​	只有在列没有包含更多信息时有用

2.Label Encoding

​	将标签转化为数字，从不(0)->有时(1)->大多数(2)->每天(3)

​	这些所有的特征变量都有明确的顺序，它们称作有序变量（**ordinal varivales**）。这些词都可以使用Label Encoding 来编码。

3.One-Hot Encoding

​	one-hot Encoding会创建新列，以指示原始数据中每个可能只的存在的值。

![](E:\Typora_Config\kaggle\xgboost\TW5m0aJ.png)

​	在，颜色有红绿蓝三种颜色，如果是红，就往红的列里面put1.

​	对比label encoding,one-hot encoding 不适用于有序变量。但是它可以很好的工作在没有明确顺序的变量。我们将这种变量称为（**nominal variables**）。

### Example

s in the previous tutorial, we will work with the [Melbourne Housing dataset](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/home).

We won't focus on the data loading step. Instead, you can imagine you are at a point where you already have the training and validation data in `X_train`, `X_valid`, `y_train`, and `y_valid`.

Output

Code



We take a peek at the training data with the `head()` method below.

In [2]:

```
X_train.head()
```

Out[2]:

|       | Type | Method | Regionname            | Rooms | Distance | Postcode | Bedroom2 | Bathroom | Landsize | Lattitude | Longtitude | Propertycount |
| :---- | :--- | :----- | :-------------------- | :---- | :------- | :------- | :------- | :------- | :------- | :-------- | :--------- | :------------ |
| 12167 | u    | S      | Southern Metropolitan | 1     | 5.0      | 3182.0   | 1.0      | 1.0      | 0.0      | -37.85984 | 144.9867   | 13240.0       |
| 6524  | h    | SA     | Western Metropolitan  | 2     | 8.0      | 3016.0   | 2.0      | 2.0      | 193.0    | -37.85800 | 144.9005   | 6380.0        |
| 8413  | h    | S      | Western Metropolitan  | 3     | 12.6     | 3020.0   | 3.0      | 1.0      | 555.0    | -37.79880 | 144.8220   | 3755.0        |
| 2919  | u    | SP     | Northern Metropolitan | 3     | 13.0     | 3046.0   | 3.0      | 1.0      | 265.0    | -37.70830 | 144.9158   | 8870.0        |
| 6043  | h    | S      | Western Metropolitan  | 3     | 13.3     | 3020.0   | 3.0      | 1.0      | 673.0    | -37.76230 | 144.8272   | 4217.0        |

Next, we obtain a list of all of the categorical variables in the training data.

We do this by checking the data type (or **dtype**) of each column. The `object` dtype indicates a column has text (there are other things it could theoretically be, but that's unimportant for our purposes). For this dataset, the columns with text indicate categorical variables.

object可以看到哪些列的值有文本

```python
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```

```python
Categorical variables:
['Type', 'Method', 'Regionname']
```

### Score from Approach 1 (Drop Categorical Variables)

We drop the `object` columns with the [`select_dtypes()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html) method.

In [5]:

```python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```



```python
MAE from Approach 1 (Drop categorical variables):
175703.48185157913
```

### Score from Approach 2 (Label Encoding)

Scikit-learn has a [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) class that can be used to get label encodings. We loop over the categorical variables and apply the label encoder separately to each column.

```python
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```

```python
MAE from Approach 2 (Label Encoding):
165936.40548390493
```

在上面的代码中，对于每一列，我们将每个唯一值随机分配给一个不同的整数。这是一种通用的方法，比提供自定义标签更简单。但是如果我们为所有有序变量提供better-informed labels ,预期的功能会进一步的提高。

##### Score from Approach 3 (One-Hot Encoding)

We use the [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) class from scikit-learn to get one-hot encodings. There are a number of parameters that can be used to customize its behavior.

- We set `handle_unknown='ignore'` to avoid errors when the validation data contains classes that aren't represented in the training data, and
- setting `sparse=False` ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

To use the encoder, we supply only the categorical columns that we want to be one-hot encoded. For instance, to encode the training data, we supply `X_train[object_cols]`. (`object_cols` in the code cell below is a list of the column names with categorical data, and so `X_train[object_cols]` contains all of the categorical data in the training set.)

```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```

```python
MAE from Approach 3 (One-Hot Encoding):
166089.4893009678
```

### 总结：

那种方法更好呢，方法1直接扔掉，MAE更高，所以最不好。但是方法2和方法3特别的接近。

In general, one-hot encoding (**Approach 3**) will typically perform best, and dropping the categorical columns (**Approach 1**) typically performs worst, but it varies on a case-by-case basis.

Conclusion[¶](https://www.kaggle.com/alexisbcook/categorical-variables#Conclusion)

The world is filled with categorical data. You will be a much more effective data scientist if you know how to use this common data type!

### 注意点：

##### 1，label encoding

在使用label encoding的时候，当测试集中的值域大于训练集中的值域，就会报错。为了处理这种情况，可以扔掉：

```python
# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
```

##### 2.one-hot encoding

 Investigating cardinality:

​	在使用one-hot 编码之前，需要先调查基数，即一列中有多少个种类的值。对于基数（cardinality）大的数据集，one-hot可能会增加很多的列。这些列，要么抛弃，要么使用label-encoding.

```python
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
```

```
[('Street', 2),
 ('Utilities', 2),
 ('CentralAir', 2),
 ('LandSlope', 3),
 ('PavedDrive', 3),
 ('LotShape', 4),
 ('LandContour', 4),
 ('ExterQual', 4),
 ('KitchenQual', 4),
 ('MSZoning', 5),
 ('LotConfig', 5),
 ('BldgType', 5),
 ('ExterCond', 5),
 ('HeatingQC', 5),
 ('Condition2', 6),
 ('RoofStyle', 6),
 ('Foundation', 6),
 ('Heating', 6),
 ('Functional', 6),
 ('SaleCondition', 6),
 ('RoofMatl', 7),
 ('HouseStyle', 8),
 ('Condition1', 9),
 ('SaleType', 9),
 ('Exterior1st', 15),
 ('Exterior2nd', 16),
 ('Neighborhood', 25)]
```

The output above shows, for each column with categorical data, the number of unique values in the column. For instance, the `'Street'` column in the training data has two unique values: `'Grvl'` and `'Pave'`, corresponding to a gravel road and a paved road, respectively.

We refer to the number of unique entries of a categorical variable as the **cardinality** of that categorical variable. For instance, the `'Street'` variable has cardinality 2.

Use the output above to answer the questions below.

```python
# Fill in the line below: How many categorical variables in the training data
# have cardinality greater than 10?
high_cardinality_numcols = 3

# Fill in the line below: How many columns are needed to one-hot encode the 
# 'Neighborhood' variable in the training data?
num_cols_neighborhood = 25

# Check your answers
step_3.a.check()
```

所以对于，one-hot encoding

```python
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
```

