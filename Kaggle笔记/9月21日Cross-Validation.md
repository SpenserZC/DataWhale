# 9月21日Cross-Validation

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
- In **Experiment 2**, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
- We repeat this process, using every fold once as the holdout set. Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if we don't use all rows simultaneously).

### 什么时候使用交叉验证

交叉验证可以更准确地衡量模型质量，如果您要做出很多建模决策，这一点尤其重要。但是运行时间可能会更长，因为它会评估多个模型（每个fold都有一个）。

因此：

​	对于较小的数据集，不需要太多的计算负担，则应运行交叉验证。对于较大的数据集，单个验证集就足够了。代码的运行速度会更快。并且您可能拥有足够的数据，因此几乎不需要重复使用其中的一些数据进行保留。

对于组成大型数据集还是小型数据集没有简单的阈值。但是，如果您的模型需要花费几分钟或更短的时间来运行，则可能值得切换到交叉验证。另外您可以运行交叉验证，看看每个实验的分数是否接近，如果每个实验产生相同的结果，则单个验证集就足够了。

# Example

We'll work with the same data as in the previous tutorial. We load the input data in `X` and the output data in `y`.

Then, we define a pipeline that uses an imputer to fill in missing values and a random forest model to make predictions.

While it's *possible* to do cross-validation without pipelines, it is quite difficult! Using a pipeline will make the code remarkably straightforward.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
```

We obtain the cross-validation scores with the [`cross_val_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) function from scikit-learn. We set the number of folds with the `cv` parameter.

```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```

```python
MAE scores:
 [301628.7893587  303164.4782723  287298.331666   236061.84754543
 260383.45111427]
```

The `scoring` parameter chooses a measure of model quality to report: in this case, we chose negative mean absolute error (MAE). The docs for scikit-learn show a [list of options](http://scikit-learn.org/stable/modules/model_evaluation.html).

It is a little surprising that we specify *negative* MAE. Scikit-learn has a convention where all metrics are defined so a high number is better. Using negatives here allows them to be consistent with that convention, though negative MAE is almost unheard of elsewhere.

We typically want a single measure of model quality to compare alternative models. So we take the average across experiments.

```python
print("Average MAE score (across experiments):")
print(scores.mean())
```

```python
Average MAE score (across experiments):
277707.3795913405
```

# Conclusion

Using cross-validation yields a much better measure of model quality, with the added benefit of cleaning up our code: note that we no longer need to keep track of separate training and validation sets. So, especially for small datasets, it's a good improvement!


  