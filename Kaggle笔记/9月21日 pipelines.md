# 9月21日 pipelines

##### 简介

Pipelines  是使数据预处理和建模代码井井有条的简单方法。具体来说，管道捆绑了预处理和建模步骤，因此您可以像使用单个捆绑包一样使用整个捆绑包。

管道的好处：

1. 更加清洁的代码：在预处理的每个步骤中考虑数据可能会变得混乱。使用管道，您无需在每个步骤中手动跟踪培训和验证数据。
2. 更少的bug：错误地使用步骤或忘记预处理步骤的机会更少。
3. 更容易实现生产：将模型从原型过渡到大规模部署的模型很难，pipelines将很有帮助。
4. 模型验证的更多选项：在下一个教程中，将会有一个实例，其中涉及到交叉验证。

##### Example

墨尔本房屋数据集

We won't focus on the data loading step. Instead, you can imagine you are at a point where you already have the training and validation data in `X_train`, `X_valid`, `y_train`, and `y_valid`.

We take a peek at the training data with the `head()` method below. Notice that the data contains both categorical data and columns with missing values. With a pipeline, it's easy to deal with both!

```
X_train.head()
```

Out[2]:

|       | Type | Method | Regionname            | Rooms | Distance | Postcode | Bedroom2 | Bathroom | Car  | Landsize | BuildingArea | YearBuilt | Lattitude | Longtitude | Propertycount |
| :---- | :--- | :----- | :-------------------- | :---- | :------- | :------- | :------- | :------- | :--- | :------- | :----------- | :-------- | :-------- | :--------- | :------------ |
| 12167 | u    | S      | Southern Metropolitan | 1     | 5.0      | 3182.0   | 1.0      | 1.0      | 1.0  | 0.0      | NaN          | 1940.0    | -37.85984 | 144.9867   | 13240.0       |
| 6524  | h    | SA     | Western Metropolitan  | 2     | 8.0      | 3016.0   | 2.0      | 2.0      | 1.0  | 193.0    | NaN          | NaN       | -37.85800 | 144.9005   | 6380.0        |
| 8413  | h    | S      | Western Metropolitan  | 3     | 12.6     | 3020.0   | 3.0      | 1.0      | 1.0  | 555.0    | NaN          | NaN       | -37.79880 | 144.8220   | 3755.0        |
| 2919  | u    | SP     | Northern Metropolitan | 3     | 13.0     | 3046.0   | 3.0      | 1.0      | 1.0  | 265.0    | NaN          | 1995.0    | -37.70830 | 144.9158   | 8870.0        |
| 6043  | h    | S      | Western Metropolitan  | 3     | 13.3     | 3020.0   | 3.0      | 1.0      | 2.0  | 673.0    | 673.0        | 1970.0    | -37.76230 | 144.8272   | 4217.0        |



We construct the full pipeline in three steps.

### Step 1: Define Preprocessing Steps

Similar to how a pipeline bundles together preprocessing and modeling steps, we use the `ColumnTransformer` class to bundle together different preprocessing steps. The code below:

- imputes missing values in **numerical** data, and
- imputes missing values and applies a one-hot encoding to **categorical** data.

  

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

##### 步骤2：定义模型

Next, we define a random forest model with the familiar [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) class.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```

##### 步骤3：创建和执行管道

最后我们用Pipeline类来定义，管道绑定的方法和模型步骤。注意：

1. 通过管道，我们可以预处理训练数据并将模型拟合到一行代码中。 （相比之下，没有管道，我们必须在单独的步骤中进行插补，一次热编码和模型训练。如果必须同时处理数字变量和分类变量，这将变得特别混乱！）
2. 通过管道，我们将X_valid中未处理的特征提供给predict（）命令，并且管道在生成预测之前自动对特征进行预处理。 （但是，在没有管道的情况下，我们必须记住在进行预测之前对验证数据进行预处理。）  

```python
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```



### Kaggle 输出

```python
# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test) # Your code here

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
```

# Cross-Validation

##### 介绍

机器学习是一个反复的过程。

你将面临选择哪些预测变量，使用哪种类型的模型，向这些模型提供哪些参数的选择。到目前为止，您已经通过数据驱动的方式通过验证模型质量来做出这些选择。

但是这种方法有一些缺点。假设有个5000行的数据集。通常，将保留20%的数据作为验证数据集或者1000行。但这在确定模型分数方面留下了一些随机的机会。就是说，即使在不同的1000行上，模型都不准确，模型在谋一组1000行上表现良好。

在极端情况下，假设只有一条数据，那么预测的结果将是100%.

通常，验证集越大，我们的模型质量度量中的随机性（噪声）就越小，并且可靠性也就越高。不幸的是，我们只能通过从训练数据中删除行来获得较大的验证集，而较小的训练数据集意味着较差的模型。

### 交叉验证

在交叉验证中，我们将模型分为不同的子数据集去衡量数据的质量。

例如，我们将数据分为5个部分，我们称之为 5-“folds”

在这之后，我们将运行一个experiment，对每个fold:

- In **Experiment 1**, we use the first fold as a validation (or holdout) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.