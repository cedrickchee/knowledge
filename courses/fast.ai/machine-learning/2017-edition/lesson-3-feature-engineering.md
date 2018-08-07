# Lesson 3 - Lesson 3 - Feature Engineering

_These are my personal notes from fast.ai machine learning course and will continue to be updated and improved if I find anything useful and relevant while I continue to review the course to study much more in-depth. Thanks for reading and happy learning!_

## Topics

* Collaborative filtering
* Favorita grocery Sales forecasting notebook
* Random forest model interpretation
  * Feature importance
* Data leakage
* Collinearity

## Lesson Resources

* [Video](https://youtu.be/YSFG_W8JxBo)
* Jupyter Notebook and code
  * [lesson2-rf_interpretation.ipynb](https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/courses/ml1/lesson2-rf_interpretation.ipynb)
* Dataset
  * [Corporación Favorita Grocery Sales Forecasting Kaggle competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

## Assignments

* Entering to Kaggle competition

## My Notes

### What is covered in today’s lesson:

**Understanding the data better by using machine learning**

- This idea is contrary to the common refrain that things like random forests are black boxes that hide meaning from us. The truth is quite the opposite. Random forests allow us to understand our data deeper and more quickly than traditional approaches.

**How to look at larger datasets**

Dataset with over 100 million rows—[Grocery Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

:question: When to use random forests [[00:02:41](https://youtu.be/YSFG_W8JxBo?t=2m41s)]?

Cannot think of anything offhand that it is definitely not going to be at least somewhat useful. So it is always worth trying. The real question might be in what situation should we try other things as well, and the short answer to that is for unstructured data (image, sound, etc), you almost certainly want to try deep learning. For collaborative filtering model (groceries competition is of that kind), neither random forest nor deep learning approach is exactly what you want and you need to do some tweaks.

### Review of last week [[00:04:42](https://youtu.be/YSFG_W8JxBo?t=4m42s)]

Reading CSV took a minute or two, and we saved it to a feather format file. Feather format is almost the same format that it lives in RAM, so it is ridiculously fast to read and write. The first thing we do is in the lesson 2 notebook is to read in the feather format file.

#### `proc_df` issue [[00:05:28](https://youtu.be/YSFG_W8JxBo?t=5m28s)]

An interesting little issue that was brought up during the week is in `proc_df` function. `proc_df` function does the following:

- Finds numeric columns which have missing values and create an additional boolean column as well as replacing the missing with medians.
- Turn the categorical objects into integer codes.

**Problem #1**: Your test set may have missing values in some columns that were not in your training set or vice versa. If that happens, you are going to get an error when you try to do the random forest since the “missing” boolean column appeared in your training set but not in the test set.

**Problem #2**: Median of the numeric value in the test set may be different from the training set. So it may process it into something which has different semantics.

**Solution**: There is now an additional return variable `nas` from `proc_df` which is a dictionary whose keys are the names of the columns that had missing values, and the values of the dictionary are the medians. Optionally, you can pass `nas` to `proc_df` as an argument to make sure that it adds those specific columns and uses those specific medians:

```python
df, y, nas = proc_df(df_raw, 'SalePrice', nas)
```

### Corporación Favorita Grocery Sales Forecasting [[00:09:25](https://youtu.be/YSFG_W8JxBo?t=9m25s)]

Let’s walk through the same process when you are working with a really large dataset. It is almost the same but there are a few cases where we cannot use the defaults because defaults run a little bit too slowly.

It is important to be able to explain the problem you are working on. The key things to understand in a machine learning problem are:

- What are the independent variables?
- What is the dependent variable (the thing you are trying to predict)?

In this competition

- Dependent variable—how many units of each kind of product were sold in each store on each day during the two week period.
- Independent variables—how many units of each product at each store on each day were sold in the last few years. For each store, where it is located and what class of store it is (metadata). For each type of product, what category of product it is, etc. For each date, we have metadata such as what the oil price was.

This is what we call a **relational dataset**. Relational dataset is one where we have a number of different pieces of information that we can join together. Specifically this kind of relational dataset is what we refer to as “star schema” where there is some central transactions table. In this competition, the central transactions table is `train.csv` which contains the number units that were sold by `date` , `store_nbr` , and `item_nbr`. From this, we can join various bits of metadata (hence the name “star” schema — there is also one called [“snowflake” schema](https://en.wikipedia.org/wiki/Snowflake_schema)).


#### Reading Data[[00:15:12](https://youtu.be/YSFG_W8JxBo?t=15m12s)]

```python
types = {'id': 'int64',
         'item_nbr': 'int32',
         'store_nbr': 'int8',
         'unit_sales': 'float32',
         'onpromotion': 'object'}
%%time
df_all = pd.read_csv(f'{PATH}train.csv', parse_dates=['date'],
                     dtype=types, infer_datetime_format=True)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
CPU times: user 1min 41s, sys: 5.08s, total: 1min 46s
Wall time: 1min 48s
```

- If you set low_me`mory=False`, it will run out of memory regardless of how much memory you have.
- In order to limit the amount of space that it takes up when you read in, we create a dictionary for each column name to the data type of that column. It is up to you to figure out the data types by running or `less` or `head` on the dataset.
- With these tweaks, we can read in 125,497,040 rows in less than 2 minutes.
- Python itself is not fast, but almost everything we want to do in Python in data science has been written for us in C or more often in Cython which is a python like language that compiles to C. In Pandas, a lot of it is written in assembly language which is heavily optimized. Behind the scene, a lot of that is going back to calling Fortran based libraries for linear algebra.

:question: Are there any performance consideration to specifying `int64` vs. `int` [[00:18:33](https://youtu.be/YSFG_W8JxBo?t=18m33s)]?

The key performance here was to use the smallest number of bits that I could to fully represent the column. If we used `int8` for `item_nbr` , the maximum `item_nbr` is bigger than 255 and it will not fit. On the other hand, if we used `int64` for the `store_nbr` , it is using more bits than necessary. Given that the whole purpose here was to avoid running out of RAM, we do not want to use up 8 times more memory than necessary. When you are working with large datasets, very often you will find that the slow piece is reading and writing to RAM, not the CPU operations. Also as a rule of thumb, smaller data types often will run faster particularly if you can use Single Instruction Multiple Data (SIMD) vectorized code, it can pack more numbers into a single vector to run at once.

Question: Do we not have to shuffle the data anymore [[00:20:11](https://youtu.be/YSFG_W8JxBo?t=20m11s)]? Although here I have read in the whole thing, when I start I never start by reading in the whole thing. By using a UNIX command `shuf`, you can get a random sample of data at the command prompt and then you can just read that. This is a good way, for example, to find out what data types to use — read in a random sample and let Pandas figure it out for you. In general, I do as much work as possible on a sample until I feel confident that I understand the sample before I move on.

To pick a random line from a file using `shuf` use the `-n` option. This limits the output to the number specified. You can also specify the output file:

`shuf -n 5 -o sample_training.csv train.csv`

`'onpromotion': ‘object'` [[00:21:28](https://youtu.be/YSFG_W8JxBo?t=21m28s)]—`object` is a general purpose Python datatype which is slow and memory heavy. The reason for this is it is a boolean which also has missing values, so we need to deal with this before we can turn it into a boolean as you see below:

```python
df_all.onpromotion.fillna(False, inplace=True)
df_all.onpromotion = df_all.onpromotion.map({'False': False,
                                             'True': True})
df_all.onpromotion = df_all.onpromotion.astype(bool)
%time df_all.to_feather('tmp/raw_groceries')
```

- `fillna(False)`: we would not do this without checking first, but some exploratory data analysis shows that it is probably an appropriate thing to do (i.e. missing means false).
- `map({‘False’: False, ‘True’: True})` : `object` usually reads in as string, so replace string `‘True’` and `‘False’` with actual booleans.
- `astype(bool)` : Then finally convert it to boolean type.
- The feather file with over 125 million records takes up something under 2.5GB of memory.
- Now it is in a nice fast format, we can save it to feather format in under 5 seconds.

Pandas is generally fast, so you can summarize every column of all 125 million records in 20 seconds:

```python
%time df_all.describe(include='all')
```

![](/images/ml_2017_lesson_3_001.png)

- First thing to look at is the dates. Dates are important because any models you put in in practice, you are going to be putting it in at some date that is later than the date that you trained it by definition. So if anything in the world changes, you need to know how your predictive accuracy changes as well. So for Kaggle or for your own project, you should always make sure that your dates do not overlap [[00:22:55](https://youtu.be/YSFG_W8JxBo?t=22m55s)].
- In this case, training set goes from 2013 to August 2017.

```python
df_test = pd.read_csv(f'{PATH}test.csv', parse_dates = ['date'],
                      dtype=types, infer_datetime_format=True)
df_test.onpromotion.fillna(False, inplace=True)
df_test.onpromotion = df_test.onpromotion.map({'False': False,
                                               'True': True})
df_test.onpromotion = df_test.onpromotion.astype(bool)
df_test.describe(include='all')
```

![](/images/ml_2017_lesson_3_002.png)

- In our test set, they go from one day later until the end of the month.
- This is a key thing — you cannot really do any useful machine learning until you understand this basic piece. You have four years of data and you are trying to predict the next two weeks. This is a fundamental thing you need to understand before you can go and do a good job at this.
- If you want to use a smaller dataset, we should use the most recent — not random set.

:question: Wouldn’t four years ago around the same time frame be important (e.g. around Christmas time)[[00:25:06](https://youtu.be/YSFG_W8JxBo?t=25m6s)]?

Exactly. It is not that there is no useful information from four years ago so we do not want to entirely throw it away. But as a first step, if you were to submit the mean, you would not submit the mean of 2012 sales, but probably want to submit the mean of last month’s sale.And later on, we might want to weight more recent dates more highly since they are probably more relevant. But we should do bunch of exploratory data analysis to check that.

```python
df_all.tail()
```

![](/images/ml_2017_lesson_3_003.png)

Here is what the bottom of the data looks like [[00:26:00](https://youtu.be/YSFG_W8JxBo?t=26m)].

```python
df_all.unit_sales = np.log1p(np.clip(df_all.unit_sales, 0, None))
```

- We have to take a log of the sales because we are trying to predict something that varies according to the ratios and they told us, in this competition, that root mean squared log error is something they care about.
- `np.clip(df_all.unit_sales, 0, None)` : there are some negative sales that represent returns and the organizer told us to consider them to be zero for the purpose of this competition. `clilp` truncates to specified min and max.
- `np.log1p` : log of the value plus 1. The competition detail tells you that they are going to use root mean squared log plus 1 error because log(0) does not make sense.

```python
%time add_datepart(df_all, 'date')

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
CPU times: user 1min 35s, sys: 16.1 s, total: 1min 51s
Wall time: 1min 53s
```

We can add date part as usual. It takes a couple of minutes, so we should run through all this on sample first to make sure it works. Once you know everything is reasonable, then go back and run on a whole set.

```python
n_valid = len(df_test)
n_trn = len(df_all) - n_valid
train, valid = split_vals(df_all, n_trn)
train.shape, valid.shape

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
((122126576, 18), (3370464, 18))
```

These lines of code are identical to what we saw for bulldozers competition. We do not need to run `train_cats` or `apply_cats` since all of the data types are already numeric (remember `apply_cats` applies the same categorical codes to validation set as the training set) [[00:27:59](https://youtu.be/YSFG_W8JxBo?t=27m59s)].

```python
%%time
trn, y, nas  = proc_df(train, 'unit_sales')
val, y_val, nas = proc_df(valid, 'unit_sales', nas)
```

Call proc_df to check the missing values and so forth.

```python
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
```

These lines of code again are identical. Then there are two changes [[00:28:48](https://youtu.be/YSFG_W8JxBo?t=28m48s)]:

```python
set_rf_samples(1_000_000)
%time x = np.array(trn, dtype=np.float32)
CPU times: user 1min 17s, sys: 18.9 s, total: 1min 36s
Wall time: 1min 37s
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=100,
                          n_jobs=8)
%time m.fit(x, y)
```

We have learned about `set_rf_samples` last week. We probably do not want to create a tree from 125 million records (not sure how long that will take). You can start with 10k or 100k and figure out how much you can run. There is no relationship between the size of the dataset and how long it takes to build the random forests — the relationship is between the number of estimators multiplied by the sample size.

:question: What is `n_job`?

In the past, it has always been `-1` [[00:29:42](https://youtu.be/YSFG_W8JxBo?t=29m42s)]. The number of jobs is the number of cores to use. I was running this on a computer that has about 60 cores and if you try to use all of them, it spent so much time spinning out jobs and it was slower. If you have lots of cores on your computer, sometimes you want less (`-1` means use every single core).

Another change was `x = np.array(trn, dtype=np.float32)`. This converts data frame into an array of floats and we fit it on that. Inside the random forest code, they do this anyway. Given that we want to run a few different random forests with a few different hyper parameters, doing this once myself saves 1 min 37 sec.

### Profiler : `%prun` [[00:30:40](https://youtu.be/YSFG_W8JxBo?t=30m40s)]

If you run a line of code that takes quite a long time, you can put `%prun` in front.

```python
%prun m.fit(x, y)
```

- This will run a profiler and tells you which lines of code took the most time. Here it was the code in scikit-learn that was the line of code that converts data frame to numpy array.
- Looking to see which things is taking up the time is called “profiling” and in software engineering is one of the most important tool. But data scientists tend to under appreciate it.
- For fun, try running `%prun` from time to time on code that takes 10–20 seconds and see if you can learn to interpret and use profiler outputs.
- Something else Jeremy noticed in the profiler is we can’t use OOB score when we do `set_rf_samples` because if we do, it will use the other 124 million rows to calculate the OOB score. Besides, we want to use the validation set that is the most recent dates rather than random.

```python
print_score(m)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
[0.7726754289860, 0.7658818632043, 0.23234198105350, 0.2193243264]
```

So this got us 0.76 validation root mean squared log error.

```python
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=10, n_jobs=8)
%time m.fit(x, y)
```

This gets us down to 0.71 even though it took a little longer.

```python
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=3, n_jobs=8)
%time m.fit(x, y)
```

This brought this error down to 0.70. `min_samples_leaf=1` did not really help. So we have a “reasonable” random forest here. But this does not give a good result on the leader board [[00:33:42](https://youtu.be/YSFG_W8JxBo?t=33m42s)]. Why? Let’s go back and see the data:

![](/images/ml_2017_lesson_3_004.png)

These are the columns we had to predict with (plus what were added by `add_datepart`). Most of the insight around how much of something you expect to sell tomorrow is likely to be wrapped up in the details about where the store is, what kind of things they tend to sell at the store, for a given item, what category of item it is. Random forest has no ability to do anything other than create binary splits on things like day of week, store number, item number. It does not know type of items or location of stores. Since its ability to understand what is going on is limited, we probably need to use the entire 4 years of data to even get some useful insights. But as soon as we start using the whole 4 years of data, a lot of the data we are using is really old. There is a Kaggle kernel that points out that what you could do is [[00:35:54](https://youtu.be/YSFG_W8JxBo?t=35m54s)]:

1. Take the last two weeks.
2. Take the average sales by store number, by item number, by on promotion, then take a mean across date.
3. Just submit that, and you come about 30th 🎉

We will talk about this in the next class, but if you can figure out how you start with that model and make it a little bit better, you will be above 30th place.

:question: Could you try to capture seasonality and trend effects by creating new columns like average sales in the month of August [[00:38:10](https://youtu.be/YSFG_W8JxBo?t=38m10s)]?

It is a great idea. The thing to figure out is how to do it because there are details to get right and are difficult- not intellectually difficult but they are difficult in a way that makes you headbutt your desk at 2am [[00:38:41](https://youtu.be/YSFG_W8JxBo?t=38m41s)].

Coding you do for machine learning is incredibly frustrating and incredibly difficult. If you get a detail wrong, much of the time it is not going to give you an exception it will just silently be slightly less good than it otherwise would have been. If you are on Kaggle, you will know that you are not doing as well as other people. But otherwise you have nothing to compare against. You will not know if your company’s model is half as good as it could be because you made a little mistake. This is why practicing on Kaggle now is great.

> You will get practice in finding all of the ways in which you can infuriatingly screw things up and you will be amazed [[00:39:38](https://youtu.be/YSFG_W8JxBo?t=39m38s)].

Even for Jeremy, there is an extraordinary array of them. As you get to get to know what they are, you will start to know how to check for them as you go. You should assume every button you press, you are going to press the wrong button. That is fine as long as you have a way to find out.

Unfortunately there is not a set of specific things you should always do, you just have to think what you know about the results of this thing I am about to do. Here is a really simple example. If you created that basic entry where you take the mean by date, by store number, by on promotion, you submitted it, and got a reasonable score. Then you think you have something that is a little bit better and you do predictions for that. How about you create a scatter plot showing the prediction of your average model on one axis versus the predictions of your new model on the other axis. You should see that they just about form a line. If they do not, then that is a very strong suggestion that you screwed something up.

:question: How often do you pull in data from other sources to supplement dataset you have [[00:41:15](https://youtu.be/YSFG_W8JxBo?t=41m15s)]?

Very often. The whole point of star schema is that you have a centric table, and you have other tables coming off it that provide metadata about it. On Kaggle, most competitions have the rule that you can use external data as long as post on the forum and is publicly available (double check the rule!). Outside of the Kaggle, you should always be looking for what external data you could possibly leverage.

:question: How about adding Ecuador’s holidays to supplement the data? [[00:42:52](https://youtu.be/YSFG_W8JxBo?t=42m52s)] That information is actually provided.

In general, one way of tackling this kind of problem is to create lots of new columns containing things like average number of sales on holidays, average percent change in sale between January and February, etc. There has been a [previous competition](https://www.kaggle.com/c/rossmann-store-sales) for a grocery chain in Germany that was almost identical. [The person who won](http://blog.kaggle.com/2015/12/21/rossmann-store-sales-winners-interview-1st-place-gert/) was a domain expert and specialist in doing logistics predictions. He created lots of columns based on his experience of what kinds of things tend to be useful for making predictions. So that is an approach that can work. The third place winner did almost no feature engineering, however, and they also had one big oversight which may have cost them the first place win. We will be learning a lot more about how to win this competition and ones like it as we go.
