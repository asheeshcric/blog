---
title: RandomForest with scikit-learn - Part 2
description: Learn how to apply a RandomForest classifier for any arbitrary dataset and generate surprisingly accurate results.
---

Welcome to the second part of the post. In this half, we will be implementing RandomForest regressor
on our own dataset. For that first, you will need to have a dataset. The one that I used is a weather
dataset from my city (Kathmandu) which is available at this [link.](https://drive.google.com/open?id=1UIZ_7VHtNhERrJkPe8DxoxuV3MmN7drT){:target="_blank"}

Once you download the dataset, you can see that there are only four columns in the dataset which are
'DATE', 'TAVG', 'TMIN', and 'TMAX'. This dataset concludes the min, max, and avg temperature for each
day from Jan-2000 to Jan-2019. For our simplicity, let's mark the max. temperature (TMAX) as our label
for the dataset, and the rest will be the features.

Now, open your jupyter notebook or any python IDE/Editor and import numpy and pandas.

{% highlight python linenos %}
    import numpy as np
    import pandas as pd
{% endhighlight %}

Use pandas to read the downloaded dataset. Make sure you put the dataset in an appropriate location
for the import

{% highlight python linenos %}
    df_raw = pd.read_csv('path/to/ktm_temps.csv',
                          parse_dates=['DATE'])
    # Checkout the dataset
    print(df_raw.head())
    print(df_raw.shape)
{% endhighlight %}
![](https://i.ibb.co/MgmZrB5/Screenshot-from-2019-03-16-21-03-57.png)

If you look closely on the csv file, you can see a lot of values from the columns are missing.
So, we need to do something for the missing values. There are certainly some good techniques to deal
with missing values when training for RandomForest, but for now we will just replace the values with
the mean of their column. For that, just type in the following command.

{% highlight python linenos %}
    df_raw.fillna(df_raw.mean(), inplace=True)
{% endhighlight %}

Now checkout the dataframe, you will see that all **NaN** values are gone for good.


### Feature Extraction from our dataset

We can see that we only have three feature columns in our dataset. In machine learning, it is generally seen that greater the
number of valuable features, better the model performs. We can see that there is a **'DATE'** column in our dataset.
We can extract this feature to add some more insights to our data. For example, instead of only
looking at the date, we can look for the day of the week or the month of the year or the day of the year. These
extracted features from one variable can make our model perform better. So, this is exactly what we
are going to do in the next step.

{% highlight python linenos %}
    # Take the 'DATE' column
    date_column = df_raw.iloc[:, 0]
    
    # Create a new dataframe with additional features
    features = pd.DataFrame({
        'year': date_column.dt.year,
        'month': date_column.dt.month,
        'dayofyear': date_column.dt.dayofyear,
        'weekofyear': date_column.dt.weekofyear,
        'dayofweek': date_column.dt.dayofweek,
    })
    
    # Add the original two features to the new dataframe
    features['TAVG'] = df_raw['TAVG']
    features['TMIN'] = df_raw['TMIN']
    
    # Check out the newly formed dataset
    features.head()
{% endhighlight %}
![](https://i.ibb.co/j5R7tNS/Screenshot-from-2019-03-16-21-16-49.png)

Since we decided the max. temperature of the day to be our label, we need to extract the labels from
the dataset. We also convert both **features** and **labels** into numpy arrays for further processing.

{% highlight python linenos %}
    labels = df_raw['TMAX']
    labels = np.array(labels)
    features = np.array(features)
{% endhighlight %}

Now, we have our dataset clean and ready. It is now time to split the dataset into training and
test sets. For this we will use an inbuilt method from sklearn called *`train_test_split`*

{% highlight python linenos %}
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.3,
                                                                                random_state=42)
{% endhighlight %}

Don't forget to checkout the shapes of and behavior of the splitted sets.

### Training the model

After having a separate training and testing set, we are now ready to train our RandomForestRegressor
model. For this, we will import the model from `sklearn.ensemble` and fit the model with the splitted
training set.


{% highlight python linenos %}
    from sklearn.ensemble import RandomForestRegressor
    
    # Instantiate the model
    # n_estimators refers to the number of crappy trees for the forest
    rf = RandomForestRegressor(n_estimators=400,
                               random_state=42)
    
    # Train the model
    rf.fit(train_features, train_labels)
{% endhighlight %}

Once the model is trained successfully, it is now time to evaluate our model and see how it performs
on our test set.

{% highlight python linenos %}
    predictions = rf.predict(test_features)
    
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    
    # Mean abs error
    mean_abs_error = round(np.mean(errors), 2)
    print(mean_abs_error)
{% endhighlight %}


{% highlight python linenos %}
    # Mean Absolute Percentage Error (MAPE)
    mape = 100 * (errors / test_labels)
    
    # Accuracy
    accuracy = 100 - np.mean(mape)
    print(accuracy)
{% endhighlight %}


In my case, the accuracy obtained for the dataset was around 96%. We can see that RandomForest performed
really well on our dataset. You can play with the hyperparameters to tune it and see how it performs
on other values. You can even use a different dataset to analyse the same model and its metrics.

You can find the complete notebook code from [here.](https://jvn.io/asheeshcric/0ca2a7d099f846e1a407b8b9310b96c4){:target="_blank"}

This concludes the second and final part of the post. Hope that you got to learn something from this
post. If you have any suggestions or queries, please feel free to comment below. I would really 
appreciate your feedback, and as always thanks for reading.