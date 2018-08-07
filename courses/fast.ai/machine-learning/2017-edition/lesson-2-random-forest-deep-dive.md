# Lesson 2 - Random Forest Deep Dive

_These are my personal notes from fast.ai machine learning course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* How random forests actually work
* Creating a good validation set
* Bagging of little bootstraps
* Ensembling
* Out-of-bag (OOB) score

## Lesson Resources

* [Video](https://youtu.be/blyXCk4sgEg)
* Jupyter Notebook and code
  * [lesson1-rf.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb)

## My Notes

For the next couple lessons, we will look at:

- how random forests actually work
- what to do if they do not work properly
- what the pros and cons are
- what we can tune
- how to interpret the result

Fastai library is a collections of best techniques to achieve state-of-the-art result. For structured data analysis, scikit-learn has a lot of great code. So what fastai does is to help us get things into scikit-learn and then interpret things out from scikit-learn.

As we noted, it is very important to deeply understand the evaluation metric [[00:06:00](https://youtu.be/blyXCk4sgEg?t=6m)].

Root Mean Squared Logarithmic Error (RMSLE):

![](/images/ml_2017_lesson_2_001.png)

![](/images/ml_2017_lesson_2_002.png)

So we took the log of the price and use root mean squared error (RMSE).

```python
df_raw.SalePrice = np.log(df_raw.SalePrice)
```

Then we made everything in the dataset to numbers by doing the following:

- add_datepart — extract date-time features Elapsed represents how many days are elapsed since January 1st, 1970.
- train_cats — converts string to pandas category data type. We then replace categorical columns with category codes by running proc_df
- proc_df also replaces missing values of the continuous columns with the median and adds the column called [column name]_na and sets it to true to indicate it was missing.







































