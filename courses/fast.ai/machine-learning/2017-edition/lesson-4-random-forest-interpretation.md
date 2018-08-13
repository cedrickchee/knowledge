# Lesson 4 - Random Forest Interpretation

_These are my personal notes from fast.ai machine learning course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Confidence based on tree variance
* Feature importance
* Logistic regression coefficients
* One-hot encoding for categorical variables
* Removing redundant features
  * Using a dendogram
  * Hierarchical clustering
  * Spearman’s R correlation matrix
* Partial dependence
  * Partial dependence plot
* What is the purpose of interpretation, what to do with that information
* Tree interpreter

## Lesson Resources

* [Video](https://youtu.be/0v93qHDqq_g)
* Jupyter Notebook and code
  * [lesson2-rf_interpretation.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/ml1/lesson2-rf_interpretation.ipynb)
* Dataset
  * [Blue Book for Bulldozers Kaggle competition](https://www.kaggle.com/c/bluebook-for-bulldozers)

## My Notes

**Terrance**: A :question: before getting started: Could we summarize the relationship between the hyper-parameters of the random forest and its effect on overfitting, dealing with collinearity, etc? [[00:01:51](https://youtu.be/0v93qHDqq_g?t=1m51s)]

Yeah, that sounds like a question born from experience. Absolutely. So, going back to [lesson 1 notebook](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb).

In terms of the hyper-parameters that are interesting, we are ignoring like pre-processing:

1. `set_rf_samples` command [[00:02:45](https://youtu.be/0v93qHDqq_g?t=2m45s)]
- Determines how many rows are in each tree. So before we start a new tree, we either bootstrap a sample (i.e. sampling with replacement from the whole thing) or we pull out a subsample of a smaller number of rows and then we build a tree from there.
- Step 1 is we have our whole big dataset, we grab a few rows at random from it, and we turn them into a smaller dataset. From that, we build a tree.

  ![](/images/ml_2017_lesson_4_001.png)

- Assuming that the tree remains balanced as we grow it, how many layers deep will this tree be (assuming we are growing it until every leaf is of size one)? log base 2 of 20K, `log2(20000)`. The depth of the tree doesn’t actually vary that much depending on the number of samples because it is related to the log of the size.
- Once we go all the way down to the bottom, how many leaf nodes would there be? 20K. We have a linear relationship between the number of leaf nodes and the size of the sample. So when you decrease the sample size, there are less final decisions that can be made. Therefore, the tree is going to be less rich in terms of what it can predict because it is making less different individual decisions and it also is making less binary choices to get to those decisions.
- Setting RF samples lower is going to mean that you overfit less, but it also means that you are going to have a less accurate individual tree model. The way Breiman, the inventor of random forest, described this is that you are trying to do two things when you build a model with bagging. One is that each individual tree/estimator is as accurate as possible (so each model is a strong predictive model). But then the across the estimators, correlation between them is as low as possible sot hat when you average them out together, you end up with something that generalizes. By decreasing the `set_rf_samples` number, we are actually decreasing the power of the estimator and increasing the correlation — so is that going to result in a better or worse validation set result for you? It depends. This is the kind of compromise which you have to figure out when you do machine learning models.

A :question: about `oob=True` [[00:06:46](https://youtu.be/0v93qHDqq_g?t=6m46s)]. All `oob=True` does is it says whatever your subsample is (it might be a bootstrap sample or a subsample), take all of the other rows (for each tree), put them into a different data set, and calculate the error on those. So it doesn’t actually impact training at all. It just gives you an additional metric which is the OOB error. So if you don’t have a validation set, then this allows you to get kind of a quasi validation set for free.

Question: If I don’t do `set_rf_samples`, what would it be called? [[00:07:55](https://youtu.be/0v93qHDqq_g?t=7m55s)] The default is, if you say `reset_rf_samples`, that causes it to bootstrap, so it will sample a new dataset as big as the original one but with replacement.

The second benefit of `set_rf_samples` is that you can run more quickly [[00:08:28](https://youtu.be/0v93qHDqq_g?t=8m28s)]. Particularly if you are running on a really large dataset like a hundred million rows, it will not be possible to run it on the full dataset. So you would either have to pick a subsample yourself before you start or you `set_rf_samples`.


2. `min_samples_leaf` [[00:08:48](https://youtu.be/0v93qHDqq_g?t=8m48s)]

Before, we assumed that `min_samples_leaf=1`, if it is set to 2, the new depth of the tree is `log2(20000)-1`. Each time we double the `min_samples_leaf` , we are removing one layer from the tree, and halving the number of leaf nodes (i.e. 10k). The result of increasing `min_samples_leaf` is that now each of our leaf nodes has more than one thing in, so we are going to get a more stable average that we are calculating in each tree. We have a little less depth (i.e. we have less decisions to make) and we have a smaller number of leaf nodes. So again, we would expect the result of that node would be that each estimator would be less predictive, but the estimators would be also less correlated. So this might help us avoid overfitting.

:question: I am not sure if every leaf node will have exactly two nodes [[00:10:33](https://youtu.be/0v93qHDqq_g?t=10m33s)]. No, it won’t necessarily have exactly two. The example of uneven split such as a leaf node containing 100 items is when they are all the same in terms of the dependent variable (suppose either, but much more likely would be the dependent). So if you get to a leaf node where every single one of them has the same auction price, or in classification every single one of them is a dog, then there is no split that you can do that’s going to improve your information. Remember, **information** is the term we use in a general sense in random forest to describe the amount of difference about the additional information we create from a split is how much we are improving the model. So you will often see this word information gain which means how much better the model got by adding an additional split point, and it could be based on RMSE or cross-entropy or how different to the standard deviations, etc.

So that is the second thing that we can do. It’s going to speed up our training because it has one less set of decisions to make. Even though there is one less set of decisions, those decisions have as much data as the previous set. So each layer of the tree can take twice as long as the previous layer. So it could definitely speed up training and generalize better.

3. `max_features` [[00:12:22](https://youtu.be/0v93qHDqq_g?t=12m22s)]

At each split, it will randomly sample columns (as opposed to `set_rf_samples` pick a subset of rows for each tree). It sounds like a small difference but it’s actually quite a different way of thinking about it. We do `set_rf_samples` so we pull out our sub sample or a bootstrap sample and that’s kept for the whole tree and we have all of the columns in there. With `max_features=0.5`, at each split, we’d pick a different half of the features. The reason we do that is because we want the trees to be as rich as possible. Particularly, if you were only doing a small number of trees (e.g. 10 trees) and you picked the same column set all the way through the tree, you are not really getting much variety in what kind of things it can find. So this way, at least in theory, seems to be something which is going to give us a better set of trees by picking a different random subset of features at every decision point.

The overall effect of the max_features is the same—it’s going to mean that each individual tree is probably going to be less accurate but the trees are going to be more varied. In particular, here this can be critical because imagine that you got one feature that is just super predictive. It’s so predictive that every random subsample you look at always starts out by splitting on that same feature then the trees are going to be very similar in the sense they all have the same initial split. But there may be some other interesting initial splits because they create different interactions of variables. So by half the time that feature won’t even be available at the top of the tree, at least half the tree are going to have a different initial split. It definitely can give us more variation and therefore it can help us to create more generalized trees that have less correlation with each other even though the individual trees probably won’t be as predictive.

![](/images/ml_2017_lesson_4_002.png)

In practice, as you add more trees, if you have `max_features=None`, that is going to use all the features every time. Then with very few trees, that can still give you a pretty good error. But as you create more trees, it’s not going to help as much because they are all pretty similar as they are all trying every single variable. Where else, if you say `max_features=sqrt` or `log2` , then as we add more estimators, we see improvements so there is an interesting interaction between those two. The chart above is from scikit-learn docs.

4. Things which do not impact our training at all [[00:16:32](https://youtu.be/0v93qHDqq_g?t=16m32s)]

`n_jobs`: simply specifies how many CPU or cores we run on, so it’ll make it faster up to a point. Generally speaking, making this more than 8 or so, they may have diminishing returns. -1 says use all of your cores. It seems weird that the default is to use one core. You will definitely get more performance by using more cores because all of you have computers with more than one core nowadays.

`oob_score=True`: This simply allows us to see OOB score. If you had set_rf_samples pretty small compared to a big dataset, OOB is going to take forever to calculate. Hopefully at some point, we will be able to fix the library so that doesn’t happen. There is no reason that need to be that way, but right now, that’s how the library works.

So they are our key basic parameters we can change [[00:17:38](https://youtu.be/0v93qHDqq_g?t=17m38s)]. There are more that you can see in the docs or `shift + tab` to have a look at them, but the ones you’ve seen are the ones that I’ve found useful to play with so feel free to play with others as well. Generally speaking, these values work well:

`max_features`: None, 0.5, sqrt, log2

`min_samples_leaf` : 1, 3, 5, 10, 25, 100… As you increase, if you notice by the time you get to 10, it’s already getting worse then there is no point going further. If you get to 100 and it’s still going better, then you can keep trying.

---

### Random Forest Interpretation [[00:18:50](https://youtu.be/0v93qHDqq_g?t=18m50s)]

Random forest interpretation is something which you could use to create some really cool Kaggle kernels. Confidence based on tree variance is something which doesn’t exist anywhere else. Feature importance definitely does and that’s already in quite a lot of Kaggle kernels. If you are looking at a competition or a dataset where nobody’s done feature importance, being the first person to do that is always going to win lots of votes because the most important thing is which features are important.

#### Confidence based on tree variance [[00:20:43](https://youtu.be/0v93qHDqq_g?t=20m43s)]

As I mentioned, when we do model interpretation, I tend to `set_rf_samples` to some subset — something small enough that I can run a model in under 10 seconds because there is no point running a super accurate model. Fifty thousand is more than enough to see each time you run an interpretation, you’ll get the same results back and so as long as that’s true, then you are already using enough data.

```python
set_rf_samples(50000)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
        max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```

#### Feature Importance [21:14]
