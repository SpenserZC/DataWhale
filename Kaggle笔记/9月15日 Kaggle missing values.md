## 9月15日 Kaggle missing values

 

### Introduction

有很多种方法处理缺失值。例如，

- 两卧室房屋的价格不包括三卧室的房屋价格；
- 调查受访者可以选择不分享他的收入

#### Three Approaches

1. A simple Option:丢掉缺失值
   1. 除非列中的大多数值丢失，模型将无法使用此发布方法访问许多（可能有用的信息）。
2. A better Option:Imputation
   1. 用一些值填入缺失值，可以是平均值。
   2. 不总是对的，但是总好过丢掉所有值
3. An Extension to Imputation
   1. 估算是标准方法。但是估算的值可能系统的高于或者低于实际值。
   2. 或者一些值在某些方面是唯一的。这样你就需要来考虑最初的值
   3. 在这种方法中，我们向以前一样，插入缺失值。但是新增一列，来标记哪些行是新插入的

### Example

In the example, we will work with the [Melbourne Housing dataset](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/home). Our model will use information such as the number of rooms and land size to predict home price.

We won't focus on the data loading step. Instead, you can imagine you are at a point where you already have the training and validation data in `X_train`, `X_valid`, `y_train`, and `y_valid`.

##### Define Function to Measure Quality of Each Approach

We define a function `score_dataset()` to compare different approaches to dealing with missing values. This function reports the [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE) from a random forest model.

##### Score from Approach 1 (Drop Columns with Missing Values)[¶](https://www.kaggle.com/alexisbcook/missing-values#Score-from-Approach-1-(Drop-Columns-with-Missing-Values))

Since we are working with both training and validation sets, we are careful to drop the same columns in both DataFrames.

```python
# Get names of columns with missing value
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```

```
MAE from Approach 1 (Drop columns with missing values):
183550.22137772635
```

##### Score from Approach 2 (Imputation)

Next, we use [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) to replace missing values with the mean value along each column.

Although it's simple, filling in the mean value generally performs quite well (but this varies by dataset). While statisticians have experimented with more complex ways to determine imputed values (such as **regression imputation**, for instance), the complex strategies typically give no additional benefit once you plug the results into sophisticated machine learning models.

```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```

```
MAE from Approach 2 (Imputation):
178166.46269899711
```

We see that **Approach 2** has lower MAE than **Approach 1**, so **Approach 2** performed better on this dataset.

##### Score from Approach 3 (An Extension to Imputation)

Next, we impute the missing values, while also keeping track of which values were imputed.

```python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```

```
MAE from Approach 3 (An Extension to Imputation):
178927.503183954
```

As we can see, **Approach 3** performed slightly worse than **Approach 2**.

##### So, why did imputation perform better than dropping the columns?

The training data has 10864 rows and 12 columns, where three columns contain missing data. For each column, less than half of the entries are missing. Thus, dropping the columns removes a lot of useful information, and so it makes sense that imputation would perform better.

```python
# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```

```
(10864, 12)
Car               49
BuildingArea    5156
YearBuilt       4307
dtype: int64
```

### 总结

​	大多数熵，填入缺失值（方法2，方法3）能够的得到更好的结果。  



#### Exercise

```python

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

