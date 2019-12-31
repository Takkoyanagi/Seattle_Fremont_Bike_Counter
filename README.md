# Seattle Fremont Bridge Hourly Bike Counter using Jupyter Notebook
*Inspired by Jake Vanderplas*

By Takaoki Koyanagi


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import seaborn as sns
import pandas as pd
```

## Created a python package to get data, with unit testing to ensure code loads properly 


```python
from Timeseries_workflow.data import get_fremont_data 
```


```python
data = get_fremont_data()
```

# Exploratory Data Analysis

## EDA: Look at weekly bike counter patterns


```python
plt.figure(figsize = (100,50))
data.resample('W').sum().plot()

```

![png](Seattle_markdown/output_7_2.png)


### Seems difficult to determine much from this plot

## EDA: Look at rolling average of days for bike counter patterns


```python
ax = data.resample('D').sum().rolling(365).sum().plot()
ax.set_ylim(0, None)
```




    (0, 1098983.95)




![png](Seattle_markdown/output_10_1.png)


### Looks like there are more counts in the East side vs. the West side starting from 2017 to 2019

## Let's group the index (time) and plot the mean to glean insight into when each side of the bridge is busy


```python
data.groupby(data.index.time).mean().plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a7f4fac898>




![png](Seattle_markdown/output_13_1.png)


### The plot suggests a commuting pattern where the West side is busy in the morning, whereas, the East side becomes busy around 4:40 pm

## Diving deeper into the data, let's create a pivot table of the total number of counts with the index being the time and columns being each day


```python
pivoted = data.pivot_table('Total', index=data.index.time, columns = data.index.date)
pivoted.iloc[:5, :5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2012-10-03</th>
      <th>2012-10-04</th>
      <th>2012-10-05</th>
      <th>2012-10-06</th>
      <th>2012-10-07</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00:00:00</th>
      <td>13.0</td>
      <td>18.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>01:00:00</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>02:00:00</th>
      <td>2.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>03:00:00</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>04:00:00</th>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivoted.plot(legend=False, alpha=0.01)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a7f5156390>




![png](Seattle_markdown/output_17_1.png)


### When we plot each day as a separate line, we can see two peaks we saw earlier that suggests a commuting pattern and another broad hump. What does this broad hump represent?

# Unsupervised learning

## Fill NA values with 0 and use PCA to reduce the dimensions to 2 to visualize


```python
X = pivoted.fillna(0).T.values
X.shape
```




    (2312, 24)




```python
X2= PCA(2, svd_solver='full').fit_transform(X)
```


```python
X2.shape
```




    (2312, 2)



### After PCA and transposing the data, we have 2 principle components and 2312 entries

## Plotting the data shows 2 clusters


```python
plt.scatter(X2[:,0], X2[:,1])
```




    <matplotlib.collections.PathCollection at 0x2a7f13584e0>




![png](Seattle_markdown/output_26_1.png)


## Using the Gaussian Mixture from sklearn, we can create a model that implements the expectation-maximization algorithm to fit the data (X). This algorithm can draw confidence ellipsoids for multivariate models (we are going to choose 2 and the default covariance-type="full").


```python
gmm = GaussianMixture(2)
gmm.fit(X)
labels = gmm.predict(X)
labels
```




    array([1, 1, 1, ..., 1, 1, 1], dtype=int64)



## Now that we have the labels of our unsupervised model, we can see if the Gaussian Mixture model was able to classify the clusters


```python
plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='plasma')
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x2a7f5622668>




![png](Seattle_markdown/output_30_1.png)


### This is exactly what we expected!! Now we can explore the two clusters further


```python
fig, ax = plt.subplots(1,2, figsize=(15,5))
plt.rcParams.update({'font.size': 55})

pivoted.T[labels == 0].T.plot(legend=False, alpha=0.1, ax=ax[0])
pivoted.T[labels == 1].T.plot(legend=False, alpha=0.1, ax=ax[1])

ax[0].set_title('Blue cluster')
ax[1].set_title('Yellow cluster')

plt.show()
```


![png](Seattle_markdown/output_32_0.png)



```python
dayofweek= pd.DatetimeIndex(pivoted.columns).dayofweek
```

## Our hypothesis was correct, the bottom right cluster is predominantly mon-fri (0-4). Whereas, the left cluster is mostly made up of sat and sun (5&6)


```python
plt.scatter(X2[:,0], X2[:,1], c=dayofweek, cmap='plasma')
plt.colorbar()

```




    <matplotlib.colorbar.Colorbar at 0x2a7f4aa42b0>




![png](Seattle_markdown/output_35_1.png)


## When we filter the "weekend" cluster to show just the weekdays, we are able to discover American holidays that falls during the weekday!! (except for 2017-02-06)


```python
dates= pd.DatetimeIndex(pivoted.columns)
dates[(labels ==0) & (dayofweek<5)]
```




    DatetimeIndex(['2012-11-22', '2012-11-23', '2012-12-24', '2012-12-25',
                   '2013-01-01', '2013-05-27', '2013-07-04', '2013-07-05',
                   '2013-09-02', '2013-11-28', '2013-11-29', '2013-12-20',
                   '2013-12-24', '2013-12-25', '2014-01-01', '2014-04-23',
                   '2014-05-26', '2014-07-04', '2014-09-01', '2014-11-27',
                   '2014-11-28', '2014-12-24', '2014-12-25', '2014-12-26',
                   '2015-01-01', '2015-05-25', '2015-07-03', '2015-09-07',
                   '2015-11-26', '2015-11-27', '2015-12-24', '2015-12-25',
                   '2016-01-01', '2016-05-30', '2016-07-04', '2016-09-05',
                   '2016-11-24', '2016-11-25', '2016-12-26', '2017-01-02',
                   '2017-02-06', '2017-05-29', '2017-07-04', '2017-09-04',
                   '2017-11-23', '2017-11-24', '2017-12-25', '2017-12-26',
                   '2018-01-01', '2018-05-28', '2018-07-04', '2018-09-03',
                   '2018-11-22', '2018-11-23', '2018-12-24', '2018-12-25',
                   '2019-01-01'],
                  dtype='datetime64[ns]', freq=None)



## Holidays and heavy snowfall (2017-02-06) [snowday](https://www.seattletimes.com/seattle-news/weather/weather-service-predicts-3-to-6-inches-of-snow-in-seattle-area/)
### Thanks for reading until the end!
-Tak Koyanagi
