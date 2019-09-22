# 9月22日XGBoost

**gradient boosting**（梯度增强）来构建和优化模型

### Introduction

在之前，学会了使用随机森林来预测而不是单个决策树。我们将随机森林成为"继承方法"。根据定义， **ensemble methods**继承方法结合了几种模型的预测（例如，有好几个树，在随机森林中）

在接下来，我们将学习另一种**ensemble methods**：gradient boosting

### gradient boosting

gradient boosting是一种通过循环将模型迭代添加到ensemble(集合)中的方法。

在开始时，集合中初始化只有一个模型，期预测可能会差很多。即使其预测非常不准确，随后对该集合的添加也将解决这些错误。

之后，我们开始这个循环：

1. First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
2. These predictions are used to calculate a loss function (like [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error), for instance).（例如均方误差）
3. Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (*Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.*)
4. Finally, we add the new model to ensemble, and ...
5. ... repeat!

In this tutorial, you will learn how to build and optimize models with **gradient boosting**. This method dominates many Kaggle competitions and achieves state-of-the-art results on a variety of datasets.

Introduction

For much of this course, you have made predictions with the random forest method, which achieves better performance than a single decision tree simply by averaging the predictions of many decision trees.

We refer to the random forest method as an "ensemble method". By definition, **ensemble methods** combine the predictions of several models (e.g., several trees, in the case of random forests).

Next, we'll learn about another ensemble method called gradient boosting.

### Gradient Boosting

**Gradient boosting** is a method that goes through cycles to iteratively add models into an ensemble.

It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

Then, we start the cycle:

- First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
- These predictions are used to calculate a loss function (like [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error), for instance).
- Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (*Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.*)
- Finally, we add the new model to ensemble, and ...
- ... repeat!

### Example

We begin by loading the training and validation data in `X_train`, `X_valid`, `y_train`, and `y_valid`.

In this example, you'll work with the XGBoost library. **XGBoost** stands for **extreme gradient boosting**, which is an implementation of gradient boosting with several additional features focused on performance and speed. (*Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.*)

In the next code cell, we import the scikit-learn API for XGBoost ([`xgboost.XGBRegressor`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)). This allows us to build and fit a model just as we would in scikit-learn. As you'll see in the output, the `XGBRegressor` class has many tunable parameters -- you'll learn about those soon!

```python
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
```

We also make predictions and evaluate the model.

```python
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```

```python
Mean Absolute Error: 280057.6586754418
```

### Parameter Tuning

XGBoost has a few parameters that can dramatically affect accuracy and training speed. The first parameters you should understand are:

`n_estimators` specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.控制模型循环多少次，它等于集合中的模型数量。

- Too *low* a value causes *underfitting*, which leads to inaccurate predictions on both training data and test data.太少了不准确
- Too *high* a value causes *overfitting*, which causes accurate predictions on training data, but inaccurate predictions on test data (*which is what we care about*).太高拟合过度，导致对训练数据的准确预测，但是对测试数据的预测不准确（这是我们所关心的）。

Typical values range from 100-1000, though this depends a lot on the `learning_rate` parameter discussed below.

Here is the code to set the number of models in the ensemble:

```python
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
```

`early_stopping_rounds` offers a way to automatically find the ideal value for `n_estimators`. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`. It's smart to set a high value for `n_estimators` and then use `early_stopping_rounds` to find the optimal time to stop iterating.

Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. Setting `early_stopping_rounds=5` is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.
  	 When using `early_stopping_rounds`, you also need to set aside some data for calculating the validation scores - this is done by setting the `eval_set` parameter.

We can modify the example above to include early stopping:

```python
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
```

If you later want to fit a model with all of your data, set `n_estimators` to whatever value you found to be optimal when run with early stopping（如果后面要用所有数据拟合模型，请将n_estimators设置为在进行早期停止运行时发现的最佳值）

`learning_rate`Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.（与其简单的将每个组件模型的预测简单的相加来做预测，我可以使用一个小数来乘它在加入之前）

This means each tree we add to the ensemble helps us less. So, we can set a higher value for `n_estimators` without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.（这意味着我们添加到集合中的每棵树对我们的帮助都会有所减少。 因此，我们可以为n_estimators设置更高的值而不会过度拟合。 如果我们使用提前停止，则会自动确定适当的树木数量。）

In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets `learning_rate=0.1`.

Modifying the example above to change the learning rate yields the following code:

```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

`n_jobs`On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter `n_jobs` equal to the number of cores on your machine. On smaller datasets, this won't help.(在考虑运行时的较大数据集上，可以使用并行性更快地构建模型。 通常将参数n_jobs设置为等于计算机上的内核数。 在较小的数据集上，这无济于事。)

The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where you would otherwise spend a long time waiting during the `fit` command.(对于计算结果没什么帮助，但是，对于数据量大的数据集，可以减少计算时间)

Here's the modified example:

```
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```

# Conclusion

[XGBoost](https://xgboost.readthedocs.io/en/latest/) is a the leading software library for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). With careful parameter tuning, you can train highly accurate models.