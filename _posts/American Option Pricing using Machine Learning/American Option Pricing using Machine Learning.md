---
layout: post
title: Pricing American Options Using Machine Learning
categories: [Project]
---


<center>Wanchaloem Wunkaew </center>
</br>
<center>leegarap@uw.edu </center>
</br>
<center>University of Washington, Seattle, WA</center>


This is an individual final project for CFRM 421/521: Machine Learning for Finance class at the University of Washington.


Note that this project is not peer-reviewed and that the project is for educational purpose. Please use it as your own risk.

The jupyter notebook of this project is accessible at <a href="https://github.com/middleOz/American-Option-Pricing-Machine-Learning/blob/main/American%20Option%20Pricing%20using%20Machine%20Learning.ipynb">this link</a>


<center><h4>Abstract</h4> </center>
</br>
We proprose machine learning methods for regression including Linear Regression, Polynomial Regression, Support Vector Regressor Ridge and Lasso Regression, Random Forest Regressor, K-Nearest Neighbors Regression, Multi-layer perceptons, and Convolutional Neural Network to price American options. The methods are trained and tested on SPDR S&P 500 ETF Trust call options, from 2021 to 2022.  The result shows that Convolutional Neural Network and the Random Forest performs better than the Binomial Tree, which we use as a benchmark, in the term of testing Root Mean Squared Errors.

## Introduction

Option pricing is one of fields in financial engineering. The formalization of option pricing methods, such as the Black-Scholes equation, has greatly impactedthe field of financial economics. Among various types of options, the American Option is a distinct financial asset that grants its holder the right to buy or sell the underlying asset at any point up to, and including, its maturity date. Unlike European options, American options do not have a closed-form solution, so it requires the use of numerical methods for their pricing. The Binomial Tree and Monte Carlo simulations are two such numerical methods capable of pricing these options. One notable limitation in all option pricing methods is the unrealistic assumptions of underlying asset price models. For instance, the Geometric Brownian motion model, which is assumed in the Black-Scholes equation, does not account for heteroskedasticity and the non-normal log return. In addition, some parameters, such as $\sigma$ in the binomial tree, in these tradtional methods/models are hard to estimate.  

In this project, we employ a range of machine learning regression models, including linear regression, polynomial regression, ridge regression, lasso regression, Support Vector Regressor, Random Forest Regressor, K-Nearest Neighbor regressor, Multilayer Perceptron regressor, and Convolutional Neural Network, to price American options. We anticipate that these models may unveil relationships between inputs (such as the strike price and stock prices from 8 days prior) and the output (the option price). As such, we expect these models to either outperform or match the performance of the traditional Binomial Tree model, which we have selected as our research benchmark. Additionally, we believe that some of these models can rectify the flaws of traditional models as outlined above.

We will divide this paper into distinct sections. In the next section, we will discuss our dataset and the Binomial Tree, which serves as our benchmark. The third section will be devoted to training machine learning models on this dataset. In the fourth section, we will summarize and discuss our findings. Lastly, in the fifth section, we will draw conclusions based on our results.

## Data Preparation and Benchmark model

We will employ the SPDR S&P 500 ETF Trust option chains from Q1 2020-Q4 2022 for our analysis. This data, which consists of more than three million options traded in markets, was downloads from <a href ="https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022">Kaggle</a>.  The dataset encompasses a wealth of information, including but not restricted to, the closing option price, the closing strike price, underlying asset price, bid and ask prices, and implied volatility. Despite the fact that the dataset includes put option data, our study will only concentrate on call options.

The features incorporated in this project consist of: strike price, dividend yield, risk-free rate, the time until the option's maturity, historical volatility, and the underlying asset (adjusted closed) prices from seven days prior (including the closed price on the date that the option is traded). A majority of these features encompass parameters used to price options in the Binomial Tree model. Note that we assume no transaction fee.

The historical volatility was calculated by the standard deviation of the logarithmic return of the underlying asset over the five years preceding the date each option was observed. The risk-free rate was obtained from that Fama-French guy website. The stock prices were obtained from Yahoo Finance via yfinance as showed below, while the dividend yield was estimated from <a href ="https://ycharts.com/companies/SPY/dividend_yield"> Ycharts</a>.

Despite the dataset's size with more than three million option data entries, we will randomly sample 100,000 options for training and testing our models due to time limitation. Also, we will choose only first 10000 of the traning data for cross validation.

The output of the model is solely the corresponding call option price, denoted as "' [C_LAST]'' within the dataset.


```python
%matplotlib inline
```


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pandas.plotting import scatter_matrix
```


```python
# get data from https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022
df = pd.read_csv("./spy_2020_2022.csv", low_memory=False)
```


```python
# Randomly chose 100000 samples for traning, validating, and testing
df = df.sample(100000,random_state = 42)
```

The columns of the option data are shown below:


```python
df.columns
```




    Index(['[QUOTE_UNIXTIME]', ' [QUOTE_READTIME]', ' [QUOTE_DATE]',
           ' [QUOTE_TIME_HOURS]', ' [UNDERLYING_LAST]', ' [EXPIRE_DATE]',
           ' [EXPIRE_UNIX]', ' [DTE]', ' [C_DELTA]', ' [C_GAMMA]', ' [C_VEGA]',
           ' [C_THETA]', ' [C_RHO]', ' [C_IV]', ' [C_VOLUME]', ' [C_LAST]',
           ' [C_SIZE]', ' [C_BID]', ' [C_ASK]', ' [STRIKE]', ' [P_BID]',
           ' [P_ASK]', ' [P_SIZE]', ' [P_LAST]', ' [P_DELTA]', ' [P_GAMMA]',
           ' [P_VEGA]', ' [P_THETA]', ' [P_RHO]', ' [P_IV]', ' [P_VOLUME]',
           ' [STRIKE_DISTANCE]', ' [STRIKE_DISTANCE_PCT]'],
          dtype='object')




```python
# load option data
#df = pd.read_csv("./spy_20_21.csv")
#df = df.iloc[:,1:]
df[' [QUOTE_DATE]'] = pd.to_datetime(df[' [QUOTE_DATE]'], format = ' %Y-%m-%d')
df.set_index(" [QUOTE_DATE]", inplace = True)
df = df[[' [UNDERLYING_LAST]',' [EXPIRE_DATE]',' [C_IV]',' [C_LAST]',' [STRIKE]']]
df.columns = ['underlying_last','maturity','implied_vol','call_last', 'K']

# load stock data
#import yfinance as yf
# spy = yf.download('SPY', '2010-01-01', '2023-02-01')
# spy.to_csv('spy.csv')

spy = pd.read_csv('./spy.csv')
spy['Date'] = pd.to_datetime(spy['Date'], format = '%Y-%m-%d')
df['maturity']=pd.to_datetime(df['maturity'], format = ' %Y-%m-%d')
spy.set_index('Date', inplace = True)
# Choose only data and adjust close
spy = spy[['Adj Close']]

# Computing log return
spy = pd.DataFrame((np.log(spy['Adj Close'].shift(-1)) - np.log(spy['Adj Close'])).dropna())
spy.columns = ['ret']
```

Compute historical volatility $q$ by finding the std of the log return 1825 days (~5 years) in back in the past.


```python
n_days_hist = 365 * 5
date_ls = list()
vol_ls = list()
spy_array = spy['ret'].to_numpy()
for i in range(n_days_hist,len(spy_array)):
    date_ls.append(spy.index[i])
    vol_ls.append(np.std(spy_array[i-n_days_hist:i+1])*np.sqrt(252))
hist_vol_df = pd.DataFrame({'Date': date_ls, "hist_vol":vol_ls})
hist_vol_df.set_index('Date', inplace = True)
```


```python
df = df.join(hist_vol_df)
```

Dividend Yields are obtained and approximated from ychart website.


```python
# Dividend Yield
# estimates from https://ycharts.com/companies/SPY/dividend_yield
# 2022 1.34%
# 2021 1.5
# 2020 1.7

d_yield = {2020: 1.7/100, 2021: 1.5/100, 2022: 1.34/100}
y_list = list()
for ind in df.index:
    y_list.append(d_yield[ind.year])
df['q'] = y_list
```

We obtain risk-free rate $r$ from <a href = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html">Kenneth French website (Fama/French 3 Factors)</a>.

We approximate the risk-free on a specific date by chooing the risk on the first day of a month corresponding to the option data.


```python
# Interest rate
# Note that the csv file below was preprocessed by removing unnecessary rows and columns that broke the read_csv
mf = pd.read_csv("F-F_Research_Data_Factors.CSV").iloc[:,:2]
mf.columns = ['Date','r']
mf['r'] /= 100
mf['Date'] = pd.to_datetime(mf['Date'], format = '%Y%m')                
mf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1926-07-01</td>
      <td>0.0296</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1926-08-01</td>
      <td>0.0264</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1926-09-01</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1926-10-01</td>
      <td>-0.0324</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1926-11-01</td>
      <td>0.0253</td>
    </tr>
  </tbody>
</table>
</div>




```python
mf = pd.read_csv("F-F_Research_Data_Factors.CSV")
mf = mf[["Unnamed: 0",'RF']]
mf.columns = ['Date','r']

r_years = mf['Date'].apply(lambda x: int(str(x)[:4]))
r_months =  mf['Date'].apply(lambda x: int(str(x)[4:]))

mf['year'] = r_years
mf['month'] = r_months
mf.drop(['Date'],axis = 1,inplace = True)
mf = mf[2020 <= mf['year']]
r_list = list()
for ind in tqdm(df.index):
    r_list.append(mf[(mf['year'] == ind.year) & (mf['month'] == ind.month)]['r'].to_numpy()[0])
df['r'] = r_list
```

    100%|█████████████████████████████████| 100000/100000 [00:42<00:00, 2368.09it/s]



```python
r_list = list()
r_i = 0

for ind in tqdm(df.index):
    while (mf.iloc[r_i,1] !=ind.year) or (mf.iloc[r_i,2]  != ind.month):
        r_i+=1
    r_list.append(mf.iloc[r_i,0])
df['r'] = r_list
```

    100%|████████████████████████████████| 100000/100000 [00:05<00:00, 18277.71it/s]


Now, we compute the time until maturity $T-t$ in a year.
This is computed by dividing a number of days between the day that the option data was observed and it maturity by 365.

We will denote this feature as $T$.



```python
# Time til maturity
dd_list = list()
for days in (df['maturity']-df.index):
    dd_list.append(days.days/365)
df['T'] = dd_list
```

A scatter matrix which conclues the dataset are shown below:


```python
scatter_matrix(df[["underlying_last","call_last", "K","hist_vol","q","r","T"]],figsize = (15,15))
plt.show()
```



![png](/assets/output_24_0.png)



### Benchmark: American option

 In the next subsection, we will discuss the traditional method of pricing American option: Binomial Tree.
Given an initial stock, in a next step, the stock price will either go up or go down, under some pre-defined multiplicative factors: u and d.
$$u = e^{\sigma \Delta t}$$
$$d = \frac{1}{u}$$
u and d depends solely on the volatility, so we used historical volatility to estimate this $\sigma$.

For stock (or index) price with continuous dividend, the risk-neutral probability is computed by
$$\hat{p} = \frac{e^{(r-q) \Delta t} - d}{u -d}$$

The price for American option at node i can be computed by
$$f_i = \max (e^{-r \Delta T} (\hat{p} f_{iu} + (1- \hat{p}) f_{id}, (s_i - K)^+)$$

where the payoff at the leaf nodes i are $ (s_i - K)^+$.

The prices of options, as observed from the market at a given time, represent the values that investors expect those options to have under the circumstances. Theorically, if we have all the necessary parameters to price an option, we can construct option prices in the market. For binomial tree, most of the parameters such as $r$, $q$,  and $S_0$ can be observed or estimated except for the $\sigma$ which is hard to estimate. One way is to construct that quantity by inversely solving the model given the option price and all other paramemters. This implied volaitlity is the volatility of the underlying asset that the market expects. When pricing option, this quantity is unknown but can be estimated. Due to volatitility smile, in many traditional model, it is hard to use the quantity in the model. Since this implied volatility can be used to reconstruct exact price of an individual option. We will not use this quantity in our model.

To estimate the volatility, werely on historical volatility of the underlying asset price. By analyzing the historical price movements of the asset, we can make an estimation of $\sigma$.


```python
# American Option pricing using binomial tree
# adapted from Kevin Mooney (see reference)

def american_call_price(S0, K, sigma, t, r = 0, q = 0, N = 3 ):

    #delta t
    t = t / (N - 1)
    u = np.exp(sigma * np.sqrt(t))
    d = 1/u

    p = (np.exp((r-q) * t) - d) / (u - d)
    stock_prices = np.zeros( (N, N) )
    call_prices = np.zeros( (N, N) )

    stock_prices[0,0] = S0
    M = 0
    for i in range(1, N ):
        M = i + 1
        stock_prices[i, 0] = d * stock_prices[i-1, 0]
        for j in range(1, M ):
            stock_prices[i, j] = u * stock_prices[i - 1, j - 1]
    expiration = stock_prices[-1,:] - K
    expiration = np.exp(-q*t *(N-1))*stock_prices[-1,:] - K
    expiration.shape = (expiration.size, )
    expiration = np.where(expiration >= 0, expiration, 0)
    call_prices[-1,:] =  expiration

    # backward computing value
    for i in range(N - 2,-1,-1):
        for j in range(i + 1):
            # American Payoff
            call_prices[i,j] = np.max([np.exp(-r * t) * ((1-p) * call_prices[i+1,j] + p * call_prices[i+1,j+1]),
                                      np.max([stock_prices[i, j] - K,0])])         
    return call_prices[0,0]
```

We use 10-step tree for option pricing.


```python
# American Option
N = 10

bm_list = list()
for i in tqdm(range(len(df))):
    current_row = df.iloc[i,:]

    S0 = current_row['underlying_last']
    K = current_row['K']
    sigma = current_row['hist_vol']
    r = current_row['r']
    q = current_row['q']
    T = current_row['T']
    bm_list.append(american_call_price(S0, K, sigma = sigma, t = T, r = r, q = q, N = N ))
df['bm'] = bm_list
```

      0%|                                    | 126/100000 [00:00<01:19, 1255.55it/s]/var/folders/6r/96ncs6hd5plcz0t1d7spstzr0000gn/T/ipykernel_47931/145875048.py:11: RuntimeWarning: invalid value encountered in double_scalars
      p = (np.exp((r-q) * t) - d) / (u - d)
    100%|█████████████████████████████████| 100000/100000 [01:12<00:00, 1377.00it/s]



```python
df = df[df['call_last'] != " "]
df['call_last'] = np.double(df['call_last'])
df.dropna(inplace = True)
```

The table for option data are shown below.
Note that we are not going to use all the columns in the table.


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>underlying_last</th>
      <th>maturity</th>
      <th>implied_vol</th>
      <th>call_last</th>
      <th>K</th>
      <th>hist_vol</th>
      <th>q</th>
      <th>r</th>
      <th>T</th>
      <th>bm</th>
    </tr>
    <tr>
      <th>[QUOTE_DATE]</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>324.87</td>
      <td>2020-09-30</td>
      <td>0.199590</td>
      <td>33.40</td>
      <td>300.0</td>
      <td>0.128190</td>
      <td>0.0170</td>
      <td>0.13</td>
      <td>0.745205</td>
      <td>47.199011</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>324.87</td>
      <td>2020-03-31</td>
      <td>0.557560</td>
      <td>100.30</td>
      <td>215.0</td>
      <td>0.128190</td>
      <td>0.0170</td>
      <td>0.13</td>
      <td>0.243836</td>
      <td>114.648588</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>324.87</td>
      <td>2020-01-27</td>
      <td>0.130940</td>
      <td>9.09</td>
      <td>316.0</td>
      <td>0.128190</td>
      <td>0.0170</td>
      <td>0.13</td>
      <td>0.068493</td>
      <td>11.905250</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>324.87</td>
      <td>2020-06-30</td>
      <td>0.286710</td>
      <td>70.85</td>
      <td>255.0</td>
      <td>0.128190</td>
      <td>0.0170</td>
      <td>0.13</td>
      <td>0.493151</td>
      <td>81.585331</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>324.87</td>
      <td>2020-01-31</td>
      <td>0.118620</td>
      <td>6.17</td>
      <td>321.5</td>
      <td>0.128190</td>
      <td>0.0170</td>
      <td>0.13</td>
      <td>0.079452</td>
      <td>8.142697</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-12-30</th>
      <td>382.44</td>
      <td>2023-01-27</td>
      <td>0.169750</td>
      <td>0.38</td>
      <td>415.0</td>
      <td>0.189685</td>
      <td>0.0134</td>
      <td>0.33</td>
      <td>0.076712</td>
      <td>1.201606</td>
    </tr>
    <tr>
      <th>2022-12-30</th>
      <td>382.44</td>
      <td>2024-01-19</td>
      <td>0.167740</td>
      <td>1.86</td>
      <td>515.0</td>
      <td>0.189685</td>
      <td>0.0134</td>
      <td>0.33</td>
      <td>1.054795</td>
      <td>27.581163</td>
    </tr>
    <tr>
      <th>2022-12-30</th>
      <td>382.44</td>
      <td>2023-01-03</td>
      <td></td>
      <td>0.00</td>
      <td>332.0</td>
      <td>0.189685</td>
      <td>0.0134</td>
      <td>0.33</td>
      <td>0.010959</td>
      <td>51.526183</td>
    </tr>
    <tr>
      <th>2022-12-30</th>
      <td>382.44</td>
      <td>2024-01-19</td>
      <td>0.193650</td>
      <td>0.16</td>
      <td>670.0</td>
      <td>0.189685</td>
      <td>0.0134</td>
      <td>0.33</td>
      <td>1.054795</td>
      <td>0.458140</td>
    </tr>
    <tr>
      <th>2022-12-30</th>
      <td>382.44</td>
      <td>2023-01-20</td>
      <td>0.681010</td>
      <td>0.01</td>
      <td>685.0</td>
      <td>0.189685</td>
      <td>0.0134</td>
      <td>0.33</td>
      <td>0.057534</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>96924 rows × 10 columns</p>
</div>



The measure for quantiative models that are widely used in machine learning is the root mean square error.
As we use the binomial model as a benchmark, we will compute the RMSE for the benchmark binomial model.


```python
# MSE
from sklearn.metrics import mean_squared_error
```


```python
np.sqrt(mean_squared_error(df['call_last'], df['bm']))
```




    51.301668827531735



The RMSE for the benchmark model is ~51 which is high.

The statistics for the absolute error is shown below. The median error is around 3.3.


```python
plt.hist(np.abs(df['bm']-df['call_last']),bins = 50, density = True)
plt.title("A histogram of error (in absolute difference) of Binomial Model")
plt.ylabel("density")
plt.xlabel("Error")
```




    Text(0.5, 0, 'Error')





![png](/assets/output_37_1.png)



## Machine Learning Model

In addition to the (closed) underlying price, days-to-maturity, historical volatility, dividend yield, interest rate and strike price, we will also use the adjusted closed stock prices 8 days lag as inputs.


```python
Tn = 8
spy = pd.read_csv('./spy.csv')
spy['Date'] = pd.to_datetime(spy['Date'], format = '%Y-%m-%d')
spy.set_index("Date",inplace= True)
spy = spy[['Adj Close']]
spy.rename(columns={"Adj Close": "t0"}, inplace = True)
spy_use = spy[spy.index >= df.index[0]]

import warnings
warnings.filterwarnings("ignore")
for i in range(1,Tn+1):
    spy_use['t-'+str(i)] = spy['t0'].shift(i).iloc[-len(spy_use['t0']):]
df = df.join(spy_use)
```


```python
# choose only numerical
df.drop(columns = ['maturity','implied_vol','bm','underlying_last'],inplace = True)
```

The first 5 rows of  final prepared dataset is shown below:


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>call_last</th>
      <th>K</th>
      <th>hist_vol</th>
      <th>q</th>
      <th>r</th>
      <th>T</th>
      <th>t0</th>
      <th>t-1</th>
      <th>t-2</th>
      <th>t-3</th>
      <th>t-4</th>
      <th>t-5</th>
      <th>t-6</th>
      <th>t-7</th>
      <th>t-8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>33.40</td>
      <td>300.0</td>
      <td>0.12819</td>
      <td>0.017</td>
      <td>0.13</td>
      <td>0.745205</td>
      <td>308.517456</td>
      <td>305.658936</td>
      <td>304.918213</td>
      <td>306.608582</td>
      <td>306.684631</td>
      <td>305.060669</td>
      <td>305.051178</td>
      <td>304.585846</td>
      <td>303.256348</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>100.30</td>
      <td>215.0</td>
      <td>0.12819</td>
      <td>0.017</td>
      <td>0.13</td>
      <td>0.243836</td>
      <td>308.517456</td>
      <td>305.658936</td>
      <td>304.918213</td>
      <td>306.608582</td>
      <td>306.684631</td>
      <td>305.060669</td>
      <td>305.051178</td>
      <td>304.585846</td>
      <td>303.256348</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>9.09</td>
      <td>316.0</td>
      <td>0.12819</td>
      <td>0.017</td>
      <td>0.13</td>
      <td>0.068493</td>
      <td>308.517456</td>
      <td>305.658936</td>
      <td>304.918213</td>
      <td>306.608582</td>
      <td>306.684631</td>
      <td>305.060669</td>
      <td>305.051178</td>
      <td>304.585846</td>
      <td>303.256348</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>70.85</td>
      <td>255.0</td>
      <td>0.12819</td>
      <td>0.017</td>
      <td>0.13</td>
      <td>0.493151</td>
      <td>308.517456</td>
      <td>305.658936</td>
      <td>304.918213</td>
      <td>306.608582</td>
      <td>306.684631</td>
      <td>305.060669</td>
      <td>305.051178</td>
      <td>304.585846</td>
      <td>303.256348</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>6.17</td>
      <td>321.5</td>
      <td>0.12819</td>
      <td>0.017</td>
      <td>0.13</td>
      <td>0.079452</td>
      <td>308.517456</td>
      <td>305.658936</td>
      <td>304.918213</td>
      <td>306.608582</td>
      <td>306.684631</td>
      <td>305.060669</td>
      <td>305.051178</td>
      <td>304.585846</td>
      <td>303.256348</td>
    </tr>
  </tbody>
</table>
</div>



The original data is timestamped with the time that each option is observed. The chronological order of the data may have an effect on the model, so it is important to take the order into account. For this option pricing project, we assume that the chronological order does not have a significant effect on the models. We make an assumption that the most chornological effects are contained in the features such as historical volatility and the stock price lags.

We shuffle and split 80% for traning set, 16% for test set, 4% for validation sets.


```python
from sklearn.model_selection import train_test_split
import datetime
# We split 80% for traning set, 16% for test set, 4% for validation sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ['call_last']),df['call_last'], test_size=0.2)
X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test, test_size=0.2)
```


```python
# If you believe the order affect, use the code below
#X_train = df[df.index <= datetime.datetime(2022,9,1)].copy()
#y_train = X_train[['call_last']]
#X_train = X_train.drop(columns = ['call_last'])

#X_test = df[df.index > datetime.datetime(2022,9,1)].copy()
#y_test = X_test[['call_last']]
#X_test = X_test.drop(columns = ['call_last'])

#X_valid = X_test.iloc[:1000,:]
#y_valid = y_test.iloc[:1000]

#X_test = X_test.iloc[1000:,:]
#y_test = y_test.iloc[1000:]
```

As we obtained the dataset, we compute the option price based on American binomial tree model on the test set in order to compare it to other models. Since a number of the test set is small, we can apply the tree for large number of steps. In this case, we use 30 steps.

### Benchmark (Binomial Tree)



```python
bm_list = list()
for i in tqdm(range(len(X_test))):
    current_row = X_test.iloc[i,:]

    S0 = current_row['t0']
    K = current_row['K']
    sigma = current_row['hist_vol']
    r = current_row['r']
    q = current_row['q']
    T = current_row['T']
    bm_list.append(american_call_price(S0, K, sigma = sigma, t = T, r = r, q = q, N = 30 ))
```

    100%|████████████████████████████████████| 15508/15508 [01:25<00:00, 180.46it/s]



```python
np.sqrt(mean_squared_error(y_test,bm_list))
```




    48.219430763590715



The root mean squared error for the binomial model is 48.219430763590715.

### Linear Regression

We use multiple linear regression with stdard scaled input.


```python
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```


```python
lin_reg = Pipeline([("std_scaler", StandardScaler()),
                     ("LinReg", LinearRegression())])
lin_reg.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;LinReg&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;LinReg&#x27;, LinearRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
np.sqrt(mean_squared_error(y_train,lin_reg.predict(X_train)))
```




    38.43095625260875




```python
np.sqrt(mean_squared_error(y_test,lin_reg.predict(X_test)))
```




    37.97049534873681



The coefficients and intercept for the model are shown below:


```python
lin_reg["LinReg"].coef_
```




    array([-37.65425523,   2.524246  ,   1.51641187,  -0.54682013,
            13.45049636,   5.76041445,   3.59107429,   3.32804942,
             3.28542487,  -0.09017812,  -0.74559882,  -1.94913388,
             0.6794828 ,   9.51963281])




```python
lin_reg["LinReg"].intercept_
```




    31.72985981248147



### Polynomial Regression

For the polynomial regression, we only consider the polynomial of degree 2 with standard scaled input. We have 14 features, and the polynomial features are going to be large.


```python
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
```


```python
PolyReg = Pipeline([("std_scaler", StandardScaler()),
          ("poly_feature", PolynomialFeatures(degree=2, include_bias=False)),
         ("LinReg",  LinearRegression())])
```


```python
PolyReg.fit(X_train, y_train)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;poly_feature&#x27;, PolynomialFeatures(include_bias=False)),
                (&#x27;LinReg&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;poly_feature&#x27;, PolynomialFeatures(include_bias=False)),
                (&#x27;LinReg&#x27;, LinearRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">PolynomialFeatures</label><div class="sk-toggleable__content"><pre>PolynomialFeatures(include_bias=False)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
# Training Loss
np.sqrt(mean_squared_error(y_train,PolyReg.predict(X_train)))
```




    36.17182222027887




```python
# Testing Loss
np.sqrt(mean_squared_error(y_test,PolyReg.predict(X_test)))
```




    36.06896946882215



### Ridge and Lasso Regression


For Ridge and Lasso Regression, we also find the best hyperparameters $\alpha$ using Grid Search.
We perform cross validation only on the first 10000 training data with 3-fold cross validation.


```python
from sklearn.linear_model import LinearRegression,Ridge,Lasso
```


```python
alpha = (0.0001, 0.001, 0.01,0.1,1,10,100,500,1000,5000,10000)
ridge_reg = Pipeline([("std_scaler", StandardScaler()),
         ("ridge",  Ridge())])
parameters = {'ridge__alpha': alpha}
rr = GridSearchCV(estimator=ridge_reg,param_grid = parameters, cv = 3)
rr.fit(X_train[:10000], y_train[:10000])

```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;ridge&#x27;, Ridge())]),
             param_grid={&#x27;ridge__alpha&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                          500, 1000, 5000, 10000)})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;ridge&#x27;, Ridge())]),
             param_grid={&#x27;ridge__alpha&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                          500, 1000, 5000, 10000)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;ridge&#x27;, Ridge())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">Ridge</label><div class="sk-toggleable__content"><pre>Ridge()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
rr.best_estimator_
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;ridge&#x27;, Ridge(alpha=10))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;ridge&#x27;, Ridge(alpha=10))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">Ridge</label><div class="sk-toggleable__content"><pre>Ridge(alpha=10)</pre></div></div></div></div></div></div></div>



We found that $\alpha = 10$ is the best alpha.
We then fit this best estimator to the whole dataset.


```python
best_ridge = rr.best_estimator_
best_ridge.fit(X_train, y_train)
```




<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;ridge&#x27;, Ridge(alpha=10))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;ridge&#x27;, Ridge(alpha=10))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label sk-toggleable__label-arrow">Ridge</label><div class="sk-toggleable__content"><pre>Ridge(alpha=10)</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
# Training Loss
np.sqrt(mean_squared_error(y_train,best_ridge.predict(X_train)))
```




    38.43095969174494




```python
# Testing Loss
np.sqrt(mean_squared_error(y_test,best_ridge.predict(X_test)))
```




    37.97019255417378



We did the same to Lasso.


```python
alpha = (0.0001, 0.001, 0.01,0.1,1,10,100,500,1000,5000,10000)
lasso_reg = Pipeline([("std_scaler", StandardScaler()),
         ("lasso",  Lasso())])
parameters = {'lasso__alpha': alpha}
lr = GridSearchCV(estimator=lasso_reg, param_grid  = parameters, cv = 3)
lr.fit(X_train[:10000], y_train[:10000])
```




<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;lasso&#x27;, Lasso())]),
             param_grid={&#x27;lasso__alpha&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                          500, 1000, 5000, 10000)})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;lasso&#x27;, Lasso())]),
             param_grid={&#x27;lasso__alpha&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                          500, 1000, 5000, 10000)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;lasso&#x27;, Lasso())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">Lasso</label><div class="sk-toggleable__content"><pre>Lasso()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
lr.best_estimator_
best_lasso = lr.best_estimator_
best_lasso
```




<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;lasso&#x27;, Lasso(alpha=0.01))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;lasso&#x27;, Lasso(alpha=0.01))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-24" type="checkbox" ><label for="sk-estimator-id-24" class="sk-toggleable__label sk-toggleable__label-arrow">Lasso</label><div class="sk-toggleable__content"><pre>Lasso(alpha=0.01)</pre></div></div></div></div></div></div></div>




```python
best_lasso.fit(X_train, y_train)
```




<style>#sk-container-id-8 {color: black;background-color: white;}#sk-container-id-8 pre{padding: 0;}#sk-container-id-8 div.sk-toggleable {background-color: white;}#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-8 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-8 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-8 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-8 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-8 div.sk-item {position: relative;z-index: 1;}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-8 div.sk-item::before, #sk-container-id-8 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-8 div.sk-label-container {text-align: center;}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-8 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;lasso&#x27;, Lasso(alpha=0.01))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-25" type="checkbox" ><label for="sk-estimator-id-25" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;lasso&#x27;, Lasso(alpha=0.01))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-26" type="checkbox" ><label for="sk-estimator-id-26" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" ><label for="sk-estimator-id-27" class="sk-toggleable__label sk-toggleable__label-arrow">Lasso</label><div class="sk-toggleable__content"><pre>Lasso(alpha=0.01)</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
# Training Loss
np.sqrt(mean_squared_error(y_train,best_lasso.predict(X_train)))
```




    38.431561232353104




```python
# Testing Loss
np.sqrt(mean_squared_error(y_test,best_lasso.predict(X_test)))
```




    37.969146928057036



The best $\alpha$ is 0.01.

We can use Lasso for feature selection as it tries to minimize unimportant features coefficients to 0.


```python
lr.best_estimator_['lasso'].coef_
```




    array([-37.64057787,   2.49377087,   1.45139028,  -0.55584586,
            13.43936913,   5.72526309,   3.68045696,   3.23670677,
             2.1807758 ,   0.        ,  -0.        ,  -0.        ,
             0.        ,   8.50465773])




```python
X_train.columns
```




    Index(['K', 'hist_vol', 'q', 'r', 'T', 't0', 't-1', 't-2', 't-3', 't-4', 't-5',
           't-6', 't-7', 't-8'],
          dtype='object')



Interestingly many of the lags of stock prices are not important.

### Support Vector Regressor

For the linear, we tune the hyperparameter C and $\epsilon$ for linear SVR.
The other setting are the same as in previous.


```python
from sklearn.svm import LinearSVR

lsvr = Pipeline([("std_scaler", StandardScaler()),
                 ('svr',LinearSVR())])
parameters = {'svr__epsilon':(0.0001, 0.001, 0.01,0.1,1,10,100,500,1000,5000,10000),
             'svr__C': (0.0001, 0.001, 0.01,0.1,1,10,100,500,1000,5000,10000)}
lsvr_gs = GridSearchCV(estimator=lsvr,param_grid = parameters, cv = 3)
lsvr_gs.fit(X_train[:10000], y_train[:10000])
```




<style>#sk-container-id-9 {color: black;background-color: white;}#sk-container-id-9 pre{padding: 0;}#sk-container-id-9 div.sk-toggleable {background-color: white;}#sk-container-id-9 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-9 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-9 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-9 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-9 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-9 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-9 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-9 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-9 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-9 div.sk-item {position: relative;z-index: 1;}#sk-container-id-9 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-9 div.sk-item::before, #sk-container-id-9 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-9 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-9 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-9 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-9 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-9 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-9 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-9 div.sk-label-container {text-align: center;}#sk-container-id-9 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-9 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;svr&#x27;, LinearSVR())]),
             param_grid={&#x27;svr__C&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500,
                                    1000, 5000, 10000),
                         &#x27;svr__epsilon&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                          500, 1000, 5000, 10000)})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" ><label for="sk-estimator-id-28" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;svr&#x27;, LinearSVR())]),
             param_grid={&#x27;svr__C&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500,
                                    1000, 5000, 10000),
                         &#x27;svr__epsilon&#x27;: (0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                          500, 1000, 5000, 10000)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" ><label for="sk-estimator-id-29" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;svr&#x27;, LinearSVR())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" ><label for="sk-estimator-id-30" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-31" type="checkbox" ><label for="sk-estimator-id-31" class="sk-toggleable__label sk-toggleable__label-arrow">LinearSVR</label><div class="sk-toggleable__content"><pre>LinearSVR()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
lsvr_gs.best_estimator_
```




<style>#sk-container-id-10 {color: black;background-color: white;}#sk-container-id-10 pre{padding: 0;}#sk-container-id-10 div.sk-toggleable {background-color: white;}#sk-container-id-10 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-10 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-10 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-10 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-10 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-10 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-10 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-10 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-10 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-10 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-10 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-10 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-10 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-10 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-10 div.sk-item {position: relative;z-index: 1;}#sk-container-id-10 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-10 div.sk-item::before, #sk-container-id-10 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-10 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-10 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-10 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-10 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-10 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-10 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-10 div.sk-label-container {text-align: center;}#sk-container-id-10 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-10 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-10" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, LinearSVR(C=10, epsilon=10))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-32" type="checkbox" ><label for="sk-estimator-id-32" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, LinearSVR(C=10, epsilon=10))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-33" type="checkbox" ><label for="sk-estimator-id-33" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-34" type="checkbox" ><label for="sk-estimator-id-34" class="sk-toggleable__label sk-toggleable__label-arrow">LinearSVR</label><div class="sk-toggleable__content"><pre>LinearSVR(C=10, epsilon=10)</pre></div></div></div></div></div></div></div>




```python
best_lsvr = lsvr_gs.best_estimator_
```


```python
best_lsvr.fit(X_train, y_train)
```




<style>#sk-container-id-11 {color: black;background-color: white;}#sk-container-id-11 pre{padding: 0;}#sk-container-id-11 div.sk-toggleable {background-color: white;}#sk-container-id-11 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-11 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-11 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-11 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-11 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-11 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-11 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-11 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-11 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-11 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-11 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-11 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-11 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-11 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-11 div.sk-item {position: relative;z-index: 1;}#sk-container-id-11 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-11 div.sk-item::before, #sk-container-id-11 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-11 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-11 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-11 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-11 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-11 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-11 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-11 div.sk-label-container {text-align: center;}#sk-container-id-11 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-11 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-11" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, LinearSVR(C=10, epsilon=10))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-35" type="checkbox" ><label for="sk-estimator-id-35" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, LinearSVR(C=10, epsilon=10))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-36" type="checkbox" ><label for="sk-estimator-id-36" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-37" type="checkbox" ><label for="sk-estimator-id-37" class="sk-toggleable__label sk-toggleable__label-arrow">LinearSVR</label><div class="sk-toggleable__content"><pre>LinearSVR(C=10, epsilon=10)</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
# Training Loss
np.sqrt(mean_squared_error(y_train,best_lsvr.predict(X_train)))
```




    38.547851070665025




```python
# Testing Loss
np.sqrt(mean_squared_error(y_test,best_lsvr.predict(X_test)))
```




    38.03050061254464



The tuned c and epsilon are 10 and 10 respectively.

Now, we consider nonlinear SVR,in addtion to $\epsilon$ and $C$, we include a type of kernel as a hyperparamter. We choose between rbf and sigmoid. Note that we reduce search space for $\epsilon$ and $C$ to accelerate the computing time.


```python
from sklearn.svm import SVR
```


```python
nlsvr = Pipeline([("std_scaler", StandardScaler()),
                 ('svr',SVR())])
parameters = {'svr__kernel':('rbf', 'sigmoid'),
              'svr__epsilon':(0.001, 0.01,0.1,1,100,5000),
             'svr__C': (0.001, 0.01,0.1,1,100,5000)
              }
nlsvr_gs = GridSearchCV(estimator=nlsvr,param_grid = parameters, cv = 3,n_jobs = -1)
nlsvr_gs.fit(X_train[:10000].to_numpy(), y_train[:10000].to_numpy().ravel())
```




<style>#sk-container-id-12 {color: black;background-color: white;}#sk-container-id-12 pre{padding: 0;}#sk-container-id-12 div.sk-toggleable {background-color: white;}#sk-container-id-12 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-12 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-12 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-12 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-12 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-12 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-12 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-12 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-12 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-12 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-12 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-12 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-12 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-12 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-12 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-12 div.sk-item {position: relative;z-index: 1;}#sk-container-id-12 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-12 div.sk-item::before, #sk-container-id-12 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-12 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-12 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-12 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-12 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-12 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-12 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-12 div.sk-label-container {text-align: center;}#sk-container-id-12 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-12 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-12" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;svr&#x27;, SVR())]),
             n_jobs=-1,
             param_grid={&#x27;svr__C&#x27;: (0.001, 0.01, 0.1, 1, 100, 5000),
                         &#x27;svr__epsilon&#x27;: (0.001, 0.01, 0.1, 1, 100, 5000),
                         &#x27;svr__kernel&#x27;: (&#x27;rbf&#x27;, &#x27;sigmoid&#x27;)})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-38" type="checkbox" ><label for="sk-estimator-id-38" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;svr&#x27;, SVR())]),
             n_jobs=-1,
             param_grid={&#x27;svr__C&#x27;: (0.001, 0.01, 0.1, 1, 100, 5000),
                         &#x27;svr__epsilon&#x27;: (0.001, 0.01, 0.1, 1, 100, 5000),
                         &#x27;svr__kernel&#x27;: (&#x27;rbf&#x27;, &#x27;sigmoid&#x27;)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-39" type="checkbox" ><label for="sk-estimator-id-39" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()), (&#x27;svr&#x27;, SVR())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-40" type="checkbox" ><label for="sk-estimator-id-40" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-41" type="checkbox" ><label for="sk-estimator-id-41" class="sk-toggleable__label sk-toggleable__label-arrow">SVR</label><div class="sk-toggleable__content"><pre>SVR()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
best_nlsvr = nlsvr_gs.best_estimator_
best_nlsvr
```




<style>#sk-container-id-13 {color: black;background-color: white;}#sk-container-id-13 pre{padding: 0;}#sk-container-id-13 div.sk-toggleable {background-color: white;}#sk-container-id-13 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-13 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-13 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-13 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-13 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-13 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-13 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-13 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-13 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-13 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-13 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-13 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-13 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-13 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-13 div.sk-item {position: relative;z-index: 1;}#sk-container-id-13 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-13 div.sk-item::before, #sk-container-id-13 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-13 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-13 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-13 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-13 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-13 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-13 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-13 div.sk-label-container {text-align: center;}#sk-container-id-13 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-13 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-13" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, SVR(C=100, epsilon=1))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-42" type="checkbox" ><label for="sk-estimator-id-42" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, SVR(C=100, epsilon=1))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-43" type="checkbox" ><label for="sk-estimator-id-43" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-44" type="checkbox" ><label for="sk-estimator-id-44" class="sk-toggleable__label sk-toggleable__label-arrow">SVR</label><div class="sk-toggleable__content"><pre>SVR(C=100, epsilon=1)</pre></div></div></div></div></div></div></div>




```python
best_nlsvr.fit(X_train, y_train)
```




<style>#sk-container-id-14 {color: black;background-color: white;}#sk-container-id-14 pre{padding: 0;}#sk-container-id-14 div.sk-toggleable {background-color: white;}#sk-container-id-14 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-14 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-14 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-14 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-14 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-14 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-14 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-14 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-14 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-14 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-14 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-14 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-14 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-14 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-14 div.sk-item {position: relative;z-index: 1;}#sk-container-id-14 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-14 div.sk-item::before, #sk-container-id-14 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-14 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-14 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-14 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-14 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-14 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-14 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-14 div.sk-label-container {text-align: center;}#sk-container-id-14 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-14 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-14" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, SVR(C=100, epsilon=1))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-45" type="checkbox" ><label for="sk-estimator-id-45" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;svr&#x27;, SVR(C=100, epsilon=1))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-46" type="checkbox" ><label for="sk-estimator-id-46" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-47" type="checkbox" ><label for="sk-estimator-id-47" class="sk-toggleable__label sk-toggleable__label-arrow">SVR</label><div class="sk-toggleable__content"><pre>SVR(C=100, epsilon=1)</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
# Training Loss
np.sqrt(mean_squared_error(y_train,best_nlsvr.predict(X_train)))
```




    37.808085641917536




```python
# Testing Loss
np.sqrt(mean_squared_error(y_test,best_nlsvr.predict(X_test)))
```




    37.92867379033997



### Random Forest


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
```

There are so many parameters, so we use random search instead of grid search. We randomly sample hyperparameters uniformly from the parameters lists for 100 samples. And we use first 10000 shuffled data entries for this tuning.


```python
rf_rg = Pipeline([('std_scaler', StandardScaler()),
                     ('rf', RandomForestRegressor())])
parameters = {'rf__n_estimators': (50,100,300,400,500),
             'rf__max_depth':(None, 8,32,64,128),
             'rf__ccp_alpha':(0,0.00000001,0.00001,0.001),
             'rf__bootstrap': [True, False]}
rf_gs =RandomizedSearchCV(estimator=rf_rg,param_distributions = parameters, cv = 3,n_jobs = -1,n_iter = 100)
```


```python
rf_gs.fit(X_train[:10000].to_numpy(), y_train[:10000].to_numpy().ravel())
```




<style>#sk-container-id-51 {color: black;background-color: white;}#sk-container-id-51 pre{padding: 0;}#sk-container-id-51 div.sk-toggleable {background-color: white;}#sk-container-id-51 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-51 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-51 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-51 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-51 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-51 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-51 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-51 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-51 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-51 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-51 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-51 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-51 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-51 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-51 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-51 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-51 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-51 div.sk-item {position: relative;z-index: 1;}#sk-container-id-51 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-51 div.sk-item::before, #sk-container-id-51 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-51 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-51 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-51 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-51 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-51 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-51 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-51 div.sk-label-container {text-align: center;}#sk-container-id-51 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-51 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-51" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=3,
                   estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                             (&#x27;rf&#x27;, RandomForestRegressor())]),
                   n_iter=160, n_jobs=-1,
                   param_distributions={&#x27;rf__bootstrap&#x27;: [True, False],
                                        &#x27;rf__ccp_alpha&#x27;: (0, 1e-08, 1e-05,
                                                          0.001),
                                        &#x27;rf__max_depth&#x27;: (None, 8, 32, 64, 128),
                                        &#x27;rf__n_estimators&#x27;: (50, 100, 300, 400,
                                                             500)})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-167" type="checkbox" ><label for="sk-estimator-id-167" class="sk-toggleable__label sk-toggleable__label-arrow">RandomizedSearchCV</label><div class="sk-toggleable__content"><pre>RandomizedSearchCV(cv=3,
                   estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                             (&#x27;rf&#x27;, RandomForestRegressor())]),
                   n_iter=160, n_jobs=-1,
                   param_distributions={&#x27;rf__bootstrap&#x27;: [True, False],
                                        &#x27;rf__ccp_alpha&#x27;: (0, 1e-08, 1e-05,
                                                          0.001),
                                        &#x27;rf__max_depth&#x27;: (None, 8, 32, 64, 128),
                                        &#x27;rf__n_estimators&#x27;: (50, 100, 300, 400,
                                                             500)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-168" type="checkbox" ><label for="sk-estimator-id-168" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;rf&#x27;, RandomForestRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-169" type="checkbox" ><label for="sk-estimator-id-169" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-170" type="checkbox" ><label for="sk-estimator-id-170" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
best_rf = rf_gs.best_estimator_
```


```python
best_rf
```




<style>#sk-container-id-52 {color: black;background-color: white;}#sk-container-id-52 pre{padding: 0;}#sk-container-id-52 div.sk-toggleable {background-color: white;}#sk-container-id-52 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-52 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-52 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-52 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-52 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-52 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-52 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-52 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-52 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-52 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-52 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-52 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-52 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-52 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-52 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-52 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-52 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-52 div.sk-item {position: relative;z-index: 1;}#sk-container-id-52 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-52 div.sk-item::before, #sk-container-id-52 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-52 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-52 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-52 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-52 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-52 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-52 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-52 div.sk-label-container {text-align: center;}#sk-container-id-52 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-52 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-52" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;rf&#x27;,
                 RandomForestRegressor(ccp_alpha=0, max_depth=8,
                                       n_estimators=400))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-171" type="checkbox" ><label for="sk-estimator-id-171" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;rf&#x27;,
                 RandomForestRegressor(ccp_alpha=0, max_depth=8,
                                       n_estimators=400))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-172" type="checkbox" ><label for="sk-estimator-id-172" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-173" type="checkbox" ><label for="sk-estimator-id-173" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(ccp_alpha=0, max_depth=8, n_estimators=400)</pre></div></div></div></div></div></div></div>




```python
best_rf.fit(X_train, y_train)
```




<style>#sk-container-id-53 {color: black;background-color: white;}#sk-container-id-53 pre{padding: 0;}#sk-container-id-53 div.sk-toggleable {background-color: white;}#sk-container-id-53 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-53 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-53 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-53 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-53 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-53 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-53 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-53 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-53 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-53 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-53 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-53 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-53 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-53 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-53 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-53 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-53 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-53 div.sk-item {position: relative;z-index: 1;}#sk-container-id-53 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-53 div.sk-item::before, #sk-container-id-53 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-53 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-53 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-53 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-53 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-53 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-53 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-53 div.sk-label-container {text-align: center;}#sk-container-id-53 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-53 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-53" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;rf&#x27;,
                 RandomForestRegressor(ccp_alpha=0, max_depth=8,
                                       n_estimators=400))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-174" type="checkbox" ><label for="sk-estimator-id-174" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;rf&#x27;,
                 RandomForestRegressor(ccp_alpha=0, max_depth=8,
                                       n_estimators=400))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-175" type="checkbox" ><label for="sk-estimator-id-175" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-176" type="checkbox" ><label for="sk-estimator-id-176" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(ccp_alpha=0, max_depth=8, n_estimators=400)</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
The root mean squared errors for training and testing set are shown below:# Training Loss
np.sqrt(mean_squared_error(y_train,best_rf.predict(X_train)))
```




    31.349066580782836




```python
#Testing RMSE
np.sqrt(mean_squared_error(y_test,best_rf.predict(X_test)))
```




    32.90546500222834



With Random Forest, we can see feature importance:


```python
best_rf['rf'].feature_importances_
```




    array([0.51617093, 0.04662939, 0.00114048, 0.00432427, 0.13790168,
           0.04360533, 0.0377016 , 0.02562464, 0.03328482, 0.01493993,
           0.05282551, 0.00782497, 0.01813578, 0.05989066])




```python
X_train.columns
# Strike Price and Time to maturity seem to be the most important
```




    Index(['K', 'hist_vol', 'q', 'r', 'T', 't0', 't-1', 't-2', 't-3', 't-4', 't-5',
           't-6', 't-7', 't-8'],
          dtype='object')



The best hyperparamters are ccp_alpha=0, max_depth=8, n_estimators=400, and bootstrap = False.

### K-Nearest Neighbors

In this model, we only tune a number of neighbors as a hyperparameter.


```python
from sklearn.neighbors import KNeighborsRegressor
```


```python
knn_rg = Pipeline([('std_scaler', StandardScaler()),
                  ('knn',KNeighborsRegressor())])
parameters = {'knn__n_neighbors': np.arange(5,100,5)}
```


```python
knn_gs = GridSearchCV(estimator=knn_rg,param_grid = parameters, cv = 3,n_jobs = -1)
knn_gs.fit(X_train[:10000], y_train[:10000])
```




<style>#sk-container-id-44 {color: black;background-color: white;}#sk-container-id-44 pre{padding: 0;}#sk-container-id-44 div.sk-toggleable {background-color: white;}#sk-container-id-44 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-44 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-44 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-44 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-44 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-44 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-44 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-44 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-44 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-44 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-44 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-44 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-44 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-44 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-44 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-44 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-44 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-44 div.sk-item {position: relative;z-index: 1;}#sk-container-id-44 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-44 div.sk-item::before, #sk-container-id-44 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-44 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-44 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-44 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-44 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-44 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-44 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-44 div.sk-label-container {text-align: center;}#sk-container-id-44 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-44 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-44" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;knn&#x27;, KNeighborsRegressor())]),
             n_jobs=-1,
             param_grid={&#x27;knn__n_neighbors&#x27;: array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
       90, 95])})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-145" type="checkbox" ><label for="sk-estimator-id-145" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=3,
             estimator=Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                                       (&#x27;knn&#x27;, KNeighborsRegressor())]),
             n_jobs=-1,
             param_grid={&#x27;knn__n_neighbors&#x27;: array([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
       90, 95])})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-146" type="checkbox" ><label for="sk-estimator-id-146" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;knn&#x27;, KNeighborsRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-147" type="checkbox" ><label for="sk-estimator-id-147" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-148" type="checkbox" ><label for="sk-estimator-id-148" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsRegressor</label><div class="sk-toggleable__content"><pre>KNeighborsRegressor()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
best_knn = knn_gs.best_estimator_
best_knn
```




<style>#sk-container-id-45 {color: black;background-color: white;}#sk-container-id-45 pre{padding: 0;}#sk-container-id-45 div.sk-toggleable {background-color: white;}#sk-container-id-45 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-45 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-45 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-45 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-45 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-45 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-45 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-45 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-45 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-45 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-45 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-45 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-45 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-45 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-45 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-45 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-45 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-45 div.sk-item {position: relative;z-index: 1;}#sk-container-id-45 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-45 div.sk-item::before, #sk-container-id-45 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-45 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-45 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-45 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-45 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-45 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-45 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-45 div.sk-label-container {text-align: center;}#sk-container-id-45 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-45 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-45" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;knn&#x27;, KNeighborsRegressor(n_neighbors=15))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-149" type="checkbox" ><label for="sk-estimator-id-149" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;knn&#x27;, KNeighborsRegressor(n_neighbors=15))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-150" type="checkbox" ><label for="sk-estimator-id-150" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-151" type="checkbox" ><label for="sk-estimator-id-151" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsRegressor</label><div class="sk-toggleable__content"><pre>KNeighborsRegressor(n_neighbors=15)</pre></div></div></div></div></div></div></div>




```python
best_knn.fit(X_train, y_train)
```




<style>#sk-container-id-55 {color: black;background-color: white;}#sk-container-id-55 pre{padding: 0;}#sk-container-id-55 div.sk-toggleable {background-color: white;}#sk-container-id-55 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-55 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-55 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-55 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-55 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-55 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-55 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-55 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-55 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-55 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-55 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-55 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-55 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-55 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-55 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-55 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-55 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-55 div.sk-item {position: relative;z-index: 1;}#sk-container-id-55 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-55 div.sk-item::before, #sk-container-id-55 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-55 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-55 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-55 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-55 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-55 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-55 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-55 div.sk-label-container {text-align: center;}#sk-container-id-55 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-55 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-55" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;knn&#x27;, KNeighborsRegressor(n_neighbors=15))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-180" type="checkbox" ><label for="sk-estimator-id-180" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;std_scaler&#x27;, StandardScaler()),
                (&#x27;knn&#x27;, KNeighborsRegressor(n_neighbors=15))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-181" type="checkbox" ><label for="sk-estimator-id-181" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-182" type="checkbox" ><label for="sk-estimator-id-182" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsRegressor</label><div class="sk-toggleable__content"><pre>KNeighborsRegressor(n_neighbors=15)</pre></div></div></div></div></div></div></div>



The root mean squared errors for training and testing set are shown below:


```python
np.sqrt(mean_squared_error(y_train,best_knn.predict(X_train)))
```




    31.92967572127278




```python
np.sqrt(mean_squared_error(y_test,best_knn.predict(X_test)))
```




    32.912277382176526



The best number of the neigbors is 15.

### Multi-layer perceptron

For neutral network, we make experiments on three models, which are multilayer perceptron (with dropout and normalization, with relu as an activation function for the hidden layers and identity for the output layer), multilayer perceptron with tuned hyperparameters, and Convolutional Neural network.


```python
from tensorflow.keras import Sequential, layers
import tensorflow as tf
```

    2023-05-31 17:15:39.270253: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(30, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(30, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation= None)
])
```


```python
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,restore_best_weights=True)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train, y_train, epochs=1000,
                    validation_data=(X_valid, y_valid),callbacks=[callback])
```

    Epoch 1/1000
    2424/2424 [==============================] - 8s 2ms/step - loss: 2348.1985 - val_loss: 1538.4142
    Epoch 2/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1858.8630 - val_loss: 1519.5778
    Epoch 3/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1826.4266 - val_loss: 1481.3267
    Epoch 4/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1821.6594 - val_loss: 1479.6355
    Epoch 5/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1815.7463 - val_loss: 1452.8606
    Epoch 6/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1803.6791 - val_loss: 1457.6923
    Epoch 7/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1796.4032 - val_loss: 1453.8772
    Epoch 8/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1796.3320 - val_loss: 1446.6718
    Epoch 9/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1788.8895 - val_loss: 1437.1597
    Epoch 10/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1762.4237 - val_loss: 1441.7910
    Epoch 11/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1789.3596 - val_loss: 1461.8259
    Epoch 12/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1782.5536 - val_loss: 1455.1730
    Epoch 13/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1772.4648 - val_loss: 1450.3599
    Epoch 14/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1753.4039 - val_loss: 1453.5958
    Epoch 15/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1778.0695 - val_loss: 1447.4929
    Epoch 16/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1759.3634 - val_loss: 1448.3613
    Epoch 17/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1760.4802 - val_loss: 1475.6681
    Epoch 18/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1776.5804 - val_loss: 1440.7992
    Epoch 19/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1775.7247 - val_loss: 1449.5691
    Epoch 20/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1755.0410 - val_loss: 1428.5800
    Epoch 21/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1759.6237 - val_loss: 1450.7557
    Epoch 22/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1749.6755 - val_loss: 1419.3580
    Epoch 23/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1746.2498 - val_loss: 1429.7271
    Epoch 24/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1759.9923 - val_loss: 1423.3920
    Epoch 25/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1757.2087 - val_loss: 1447.6754
    Epoch 26/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1750.8146 - val_loss: 1440.7625
    Epoch 27/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1752.6371 - val_loss: 1438.2209
    Epoch 28/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1754.3665 - val_loss: 1447.5432
    Epoch 29/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1750.0234 - val_loss: 1419.1761
    Epoch 30/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1746.3068 - val_loss: 1438.0330
    Epoch 31/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1737.3093 - val_loss: 1446.1461
    Epoch 32/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1750.8802 - val_loss: 1448.6462
    Epoch 33/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1751.4987 - val_loss: 1429.7606
    Epoch 34/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1759.7916 - val_loss: 1446.9124
    Epoch 35/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1725.5071 - val_loss: 1451.0299
    Epoch 36/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1739.6782 - val_loss: 1439.0204
    Epoch 37/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1745.8899 - val_loss: 1427.5814
    Epoch 38/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1742.3463 - val_loss: 1435.0360
    Epoch 39/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1738.2543 - val_loss: 1416.4845
    Epoch 40/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1735.5444 - val_loss: 1423.9437
    Epoch 41/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1763.2434 - val_loss: 1444.3976
    Epoch 42/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1735.7745 - val_loss: 1404.2688
    Epoch 43/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1725.9673 - val_loss: 1435.0131
    Epoch 44/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1732.7388 - val_loss: 1407.1790
    Epoch 45/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1742.6547 - val_loss: 1428.5132
    Epoch 46/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1735.9810 - val_loss: 1442.2273
    Epoch 47/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1729.7102 - val_loss: 1431.0380
    Epoch 48/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1746.6910 - val_loss: 1427.7258
    Epoch 49/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1743.8966 - val_loss: 1440.8820
    Epoch 50/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1743.2539 - val_loss: 1424.8864
    Epoch 51/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1716.4867 - val_loss: 1463.6836
    Epoch 52/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1732.3893 - val_loss: 1475.7570
    Epoch 53/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1732.6115 - val_loss: 1433.2356
    Epoch 54/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1727.6232 - val_loss: 1427.9457
    Epoch 55/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1735.5585 - val_loss: 1421.7649
    Epoch 56/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1741.6429 - val_loss: 1435.6066
    Epoch 57/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1699.8602 - val_loss: 1439.7230
    Epoch 58/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1718.7030 - val_loss: 1394.6166
    Epoch 59/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1732.8438 - val_loss: 1426.4042
    Epoch 60/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1740.5402 - val_loss: 1432.9884
    Epoch 61/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1731.2108 - val_loss: 1425.0367
    Epoch 62/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1729.8733 - val_loss: 1416.9202
    Epoch 63/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1734.1432 - val_loss: 1434.6796
    Epoch 64/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1737.9097 - val_loss: 1417.3958
    Epoch 65/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1734.9419 - val_loss: 1415.4495
    Epoch 66/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1721.2323 - val_loss: 1440.2329
    Epoch 67/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1706.9197 - val_loss: 1414.5059
    Epoch 68/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1725.6825 - val_loss: 1426.8889
    Epoch 69/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1736.7288 - val_loss: 1434.6287
    Epoch 70/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1727.9434 - val_loss: 1432.4276
    Epoch 71/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1720.5653 - val_loss: 1415.3933
    Epoch 72/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1728.3397 - val_loss: 1405.6853
    Epoch 73/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1732.8448 - val_loss: 1428.6942
    Epoch 74/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1720.3589 - val_loss: 1417.5557
    Epoch 75/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1711.6993 - val_loss: 1435.7582
    Epoch 76/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1717.6664 - val_loss: 1458.3612
    Epoch 77/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1722.2780 - val_loss: 1415.3036
    Epoch 78/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1716.8831 - val_loss: 1432.3440
    Epoch 79/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1728.6538 - val_loss: 1426.1598
    Epoch 80/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1713.8955 - val_loss: 1415.4485
    Epoch 81/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1717.9971 - val_loss: 1428.0583
    Epoch 82/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1729.1248 - val_loss: 1423.9069
    Epoch 83/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1726.0729 - val_loss: 1415.2335
    Epoch 84/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1720.4703 - val_loss: 1414.7393
    Epoch 85/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1731.6072 - val_loss: 1408.4889
    Epoch 86/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1702.9434 - val_loss: 1398.9899
    Epoch 87/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1727.0245 - val_loss: 1415.9796
    Epoch 88/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1728.9961 - val_loss: 1432.7274
    Epoch 89/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1720.5521 - val_loss: 1440.2959
    Epoch 90/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1729.8057 - val_loss: 1445.4159
    Epoch 91/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1708.5405 - val_loss: 1454.3297
    Epoch 92/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1726.8213 - val_loss: 1417.4740
    Epoch 93/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1720.4965 - val_loss: 1422.0383
    Epoch 94/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1699.9498 - val_loss: 1419.1676
    Epoch 95/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1710.6783 - val_loss: 1401.8184
    Epoch 96/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1719.1256 - val_loss: 1424.0421
    Epoch 97/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1717.6489 - val_loss: 1425.7606
    Epoch 98/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1712.7775 - val_loss: 1424.1932
    Epoch 99/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1713.8696 - val_loss: 1408.1414
    Epoch 100/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1724.4841 - val_loss: 1422.6249
    Epoch 101/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1701.7123 - val_loss: 1427.1375
    Epoch 102/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1726.4032 - val_loss: 1447.1965
    Epoch 103/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1717.5742 - val_loss: 1432.3199
    Epoch 104/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1725.1085 - val_loss: 1471.1893
    Epoch 105/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1729.5525 - val_loss: 1410.8346
    Epoch 106/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1699.8079 - val_loss: 1426.6115
    Epoch 107/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1737.1375 - val_loss: 1446.8939
    Epoch 108/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1710.6053 - val_loss: 1421.2778



```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     batch_normalization (BatchN  (None, 14)               56        
     ormalization)                                                   

     dropout (Dropout)           (None, 14)                0         

     dense (Dense)               (None, 30)                450       

     batch_normalization_1 (Batc  (None, 30)               120       
     hNormalization)                                                 

     dropout_1 (Dropout)         (None, 30)                0         

     dense_1 (Dense)             (None, 30)                930       

     batch_normalization_2 (Batc  (None, 30)               120       
     hNormalization)                                                 

     dropout_2 (Dropout)         (None, 30)                0         

     dense_2 (Dense)             (None, 20)                620       

     batch_normalization_3 (Batc  (None, 20)               80        
     hNormalization)                                                 

     dropout_3 (Dropout)         (None, 20)                0         

     dense_3 (Dense)             (None, 10)                210       

     batch_normalization_4 (Batc  (None, 10)               40        
     hNormalization)                                                 

     dropout_4 (Dropout)         (None, 10)                0         

     dense_4 (Dense)             (None, 1)                 11        

    =================================================================
    Total params: 2,637
    Trainable params: 2,429
    Non-trainable params: 208
    _________________________________________________________________


The learning curve is shown below:


```python
plt.plot(history.history['loss'],label = "loss")
plt.plot(history.history['val_loss'], label = "val_loss")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epoch")
```




    Text(0.5, 0, 'Epoch')





![png](/assets/output_141_1.png)



The root mean squared errors for training and testing set are shown below:


```python
#trainig RMSE
np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
```

    2424/2424 [==============================] - 2s 714us/step





    36.21230441027238




```python
#testing RMSE
np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
```

    485/485 [==============================] - 0s 675us/step





    35.71456726735933



The model seems to be underfitting.

### Hyperparamter-tuned MLP

We use keras tuner library to tune the hyperparameters of the network. The hyperparameter space is shown below.


```python
import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=1, max_value=8)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    l2_rate = hp.Float("l2", min_value=1e-4, max_value=100,
                             sampling="log")
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Normalization(input_shape=X_train.shape[1:]))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu",kernel_initializer="he_normal",kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.add(tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.compile(loss="mse", optimizer=optimizer)
    return model
```


```python
random_search_tuner = kt.RandomSearch(
    build_model, objective="val_loss", max_trials=20, seed=42)
random_search_tuner.search(X_train[:5000], y_train[:5000], epochs=150,
                           validation_data=(X_valid, y_valid))

```

    Trial 20 Complete [00h 00m 53s]
    val_loss: 1461.2305908203125

    Best val_loss So Far: 1451.614013671875
    Total elapsed time: 00h 15m 31s
    INFO:tensorflow:Oracle triggered exit



```python
random_search_tuner.get_best_hyperparameters()[0].values
```




    {'n_hidden': 7,
     'n_neurons': 15,
     'learning_rate': 0.0006237028864858578,
     'l2': 0.0003065801184974072}



The best hyperparameters we found are 7 hidden layers with 15 neurons each, 0.0006237028864858578 learning rate, and l2 = 0.0003065801184974072 for the l2 regularization.


```python
best_nn = random_search_tuner.get_best_models()[0]
```


```python
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100,restore_best_weights=True)
history_best_nn = best_nn.fit(X_train, y_train, epochs=1000,
                    validation_data=(X_valid, y_valid),callbacks=[callback])
```

    Epoch 1/1000
    2424/2424 [==============================] - 6s 1ms/step - loss: 1364.6182 - val_loss: 1498.1863
    Epoch 2/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1363.1213 - val_loss: 1462.6725
    Epoch 3/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1354.9094 - val_loss: 1493.1799
    Epoch 4/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1349.7559 - val_loss: 1452.7467
    Epoch 5/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1347.2516 - val_loss: 1499.4685
    Epoch 6/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1341.6652 - val_loss: 1437.6245
    Epoch 7/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1340.0056 - val_loss: 1423.4025
    Epoch 8/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1340.5402 - val_loss: 1513.4679
    Epoch 9/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1340.7520 - val_loss: 1569.9033
    Epoch 10/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1341.2555 - val_loss: 1429.2285
    Epoch 11/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1336.1398 - val_loss: 1427.5570
    Epoch 12/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1335.7197 - val_loss: 1418.7195
    Epoch 13/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1328.2504 - val_loss: 1425.6616
    Epoch 14/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1330.4080 - val_loss: 1400.9790
    Epoch 15/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1333.8114 - val_loss: 1433.3483
    Epoch 16/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1327.5535 - val_loss: 1425.0586
    Epoch 17/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1322.5675 - val_loss: 1468.2742
    Epoch 18/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1328.0420 - val_loss: 1401.6270
    Epoch 19/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1321.7260 - val_loss: 1409.4268
    Epoch 20/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1322.4156 - val_loss: 1471.2164
    Epoch 21/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1323.3676 - val_loss: 1419.7789
    Epoch 22/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1317.7885 - val_loss: 1413.2256
    Epoch 23/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1319.0558 - val_loss: 1434.8601
    Epoch 24/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1318.9410 - val_loss: 1422.4067
    Epoch 25/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1317.8682 - val_loss: 1484.2041
    Epoch 26/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1310.8678 - val_loss: 1508.7574
    Epoch 27/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1309.4401 - val_loss: 1460.7371
    Epoch 28/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1317.2145 - val_loss: 1484.6249
    Epoch 29/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1315.0941 - val_loss: 1417.9153
    Epoch 30/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1307.3124 - val_loss: 1383.6708
    Epoch 31/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1311.8610 - val_loss: 1386.5297
    Epoch 32/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1316.4789 - val_loss: 1409.3855
    Epoch 33/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1306.3677 - val_loss: 1423.1759
    Epoch 34/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1307.3098 - val_loss: 1424.1090
    Epoch 35/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1309.0583 - val_loss: 1484.6648
    Epoch 36/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1311.3448 - val_loss: 1406.2308
    Epoch 37/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1310.5833 - val_loss: 1396.6992
    Epoch 38/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1306.8617 - val_loss: 1401.8198
    Epoch 39/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1306.9996 - val_loss: 1387.1191
    Epoch 40/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1303.6805 - val_loss: 1395.5427
    Epoch 41/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1307.1783 - val_loss: 1493.2867
    Epoch 42/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1300.9042 - val_loss: 1430.4606
    Epoch 43/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1308.8569 - val_loss: 1416.1036
    Epoch 44/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1308.3793 - val_loss: 1385.8787
    Epoch 45/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1303.3976 - val_loss: 1386.7449
    Epoch 46/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1304.5983 - val_loss: 1444.5226
    Epoch 47/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1303.4165 - val_loss: 1416.3625
    Epoch 48/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1305.6034 - val_loss: 1403.2960
    Epoch 49/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1303.5551 - val_loss: 1470.1138
    Epoch 50/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1305.3503 - val_loss: 1449.6703
    Epoch 51/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1299.5664 - val_loss: 1413.4106
    Epoch 52/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1300.1431 - val_loss: 1383.1802
    Epoch 53/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1298.5760 - val_loss: 1407.6246
    Epoch 54/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1297.3755 - val_loss: 1418.1981
    Epoch 55/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1309.8547 - val_loss: 1396.7052
    Epoch 56/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1299.8823 - val_loss: 1411.4867
    Epoch 57/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1299.6304 - val_loss: 1382.4707
    Epoch 58/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1297.1543 - val_loss: 1408.1846
    Epoch 59/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1299.1917 - val_loss: 1382.2891
    Epoch 60/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1300.4590 - val_loss: 1377.1444
    Epoch 61/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1296.9580 - val_loss: 1414.5353
    Epoch 62/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1295.5001 - val_loss: 1410.3369
    Epoch 63/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1295.6877 - val_loss: 1474.8268
    Epoch 64/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1296.8936 - val_loss: 1462.0864
    Epoch 65/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1292.2124 - val_loss: 1392.4818
    Epoch 66/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1299.3850 - val_loss: 1385.2305
    Epoch 67/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1297.6859 - val_loss: 1376.8387
    Epoch 68/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1293.0333 - val_loss: 1496.0873
    Epoch 69/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1291.7650 - val_loss: 1383.3090
    Epoch 70/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1292.5830 - val_loss: 1425.5532
    Epoch 71/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1290.5063 - val_loss: 1418.2140
    Epoch 72/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1298.6997 - val_loss: 1372.8307
    Epoch 73/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1291.1085 - val_loss: 1404.3828
    Epoch 74/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1294.0791 - val_loss: 1398.5275
    Epoch 75/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1294.1788 - val_loss: 1419.4735
    Epoch 76/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1288.8263 - val_loss: 1387.4668
    Epoch 77/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1291.7236 - val_loss: 1449.1144
    Epoch 78/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1293.4061 - val_loss: 1362.5378
    Epoch 79/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1289.6440 - val_loss: 1497.2910
    Epoch 80/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1291.5959 - val_loss: 1411.2170
    Epoch 81/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1289.1887 - val_loss: 1416.5708
    Epoch 82/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1285.9327 - val_loss: 1476.6334
    Epoch 83/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1287.9426 - val_loss: 1393.4445
    Epoch 84/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1291.1870 - val_loss: 1381.0616
    Epoch 85/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1292.0215 - val_loss: 1375.1881
    Epoch 86/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1302.5002 - val_loss: 1413.6367
    Epoch 87/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1304.4825 - val_loss: 1413.9423
    Epoch 88/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1301.3004 - val_loss: 1396.5070
    Epoch 89/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1297.1068 - val_loss: 1403.4583
    Epoch 90/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1295.5410 - val_loss: 1444.8091
    Epoch 91/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1296.8639 - val_loss: 1642.1996
    Epoch 92/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1298.0598 - val_loss: 1383.2399
    Epoch 93/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1293.7683 - val_loss: 1400.7040
    Epoch 94/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1295.6245 - val_loss: 1405.1207
    Epoch 95/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1295.0397 - val_loss: 1393.5569
    Epoch 96/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1290.4719 - val_loss: 1413.7733
    Epoch 97/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1292.6874 - val_loss: 1401.7083
    Epoch 98/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1288.6898 - val_loss: 1377.9130
    Epoch 99/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1291.7607 - val_loss: 1378.2542
    Epoch 100/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1290.3800 - val_loss: 1417.4948
    Epoch 101/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1292.4379 - val_loss: 1399.3606
    Epoch 102/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1287.6709 - val_loss: 1473.5056
    Epoch 103/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1290.6011 - val_loss: 1388.3289
    Epoch 104/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1288.6428 - val_loss: 1454.5400
    Epoch 105/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1289.4052 - val_loss: 1415.3314
    Epoch 106/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1290.4834 - val_loss: 1391.7068
    Epoch 107/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1288.2966 - val_loss: 1462.8899
    Epoch 108/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1287.3923 - val_loss: 1386.7235
    Epoch 109/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1288.1456 - val_loss: 1416.2400
    Epoch 110/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1285.1681 - val_loss: 1445.7714
    Epoch 111/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1289.7953 - val_loss: 1382.6591
    Epoch 112/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1283.0540 - val_loss: 1409.8440
    Epoch 113/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1286.6077 - val_loss: 1401.5734
    Epoch 114/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1287.9523 - val_loss: 1370.2687
    Epoch 115/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1283.5190 - val_loss: 1380.1663
    Epoch 116/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1289.4071 - val_loss: 1428.0024
    Epoch 117/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1280.5973 - val_loss: 1387.0465
    Epoch 118/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1282.8215 - val_loss: 1436.8264
    Epoch 119/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.5298 - val_loss: 1372.1311
    Epoch 120/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.3448 - val_loss: 1452.7546
    Epoch 121/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1284.1759 - val_loss: 1363.5353
    Epoch 122/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1282.9178 - val_loss: 1381.4111
    Epoch 123/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1282.4863 - val_loss: 1405.8273
    Epoch 124/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.2382 - val_loss: 1366.9683
    Epoch 125/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1278.4465 - val_loss: 1386.6677
    Epoch 126/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1277.6810 - val_loss: 1377.4130
    Epoch 127/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1280.3582 - val_loss: 1389.1605
    Epoch 128/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.3627 - val_loss: 1634.7262
    Epoch 129/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.1450 - val_loss: 1449.2158
    Epoch 130/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1282.5782 - val_loss: 1378.3988
    Epoch 131/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1279.5955 - val_loss: 1369.0695
    Epoch 132/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1282.0540 - val_loss: 1390.4357
    Epoch 133/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1283.5043 - val_loss: 1441.0704
    Epoch 134/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1278.8175 - val_loss: 1393.5648
    Epoch 135/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1279.4724 - val_loss: 1533.0723
    Epoch 136/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1293.1829 - val_loss: 1408.5120
    Epoch 137/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1279.3025 - val_loss: 1368.6935
    Epoch 138/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1280.2751 - val_loss: 1393.4186
    Epoch 139/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1278.2380 - val_loss: 1418.6953
    Epoch 140/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1280.1469 - val_loss: 1400.7045
    Epoch 141/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.8773 - val_loss: 1463.9827
    Epoch 142/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1290.2437 - val_loss: 1441.4587
    Epoch 143/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1291.1470 - val_loss: 1390.3928
    Epoch 144/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1277.0847 - val_loss: 1379.5386
    Epoch 145/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1274.8203 - val_loss: 1427.4891
    Epoch 146/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1278.0580 - val_loss: 1485.2283
    Epoch 147/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.2335 - val_loss: 1397.7173
    Epoch 148/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1279.4534 - val_loss: 1388.6392
    Epoch 149/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1276.0879 - val_loss: 1369.0419
    Epoch 150/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.5133 - val_loss: 1409.0029
    Epoch 151/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1278.4556 - val_loss: 1371.6171
    Epoch 152/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1276.3888 - val_loss: 1375.8518
    Epoch 153/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1277.9033 - val_loss: 1367.7881
    Epoch 154/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1274.0396 - val_loss: 1463.5127
    Epoch 155/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1276.9036 - val_loss: 1375.0033
    Epoch 156/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1277.3983 - val_loss: 1434.8405
    Epoch 157/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1277.0344 - val_loss: 1427.3362
    Epoch 158/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.9120 - val_loss: 1360.6544
    Epoch 159/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1279.3248 - val_loss: 1361.7107
    Epoch 160/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1277.2719 - val_loss: 1397.4290
    Epoch 161/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1275.5708 - val_loss: 1379.9587
    Epoch 162/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1280.3903 - val_loss: 1364.7113
    Epoch 163/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1278.4695 - val_loss: 1469.0111
    Epoch 164/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1281.9558 - val_loss: 1397.5836
    Epoch 165/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1274.2274 - val_loss: 1489.8499
    Epoch 166/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1276.2310 - val_loss: 1371.2703
    Epoch 167/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.4867 - val_loss: 1388.2601
    Epoch 168/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1274.9396 - val_loss: 1364.3513
    Epoch 169/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1272.7573 - val_loss: 1448.8358
    Epoch 170/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1280.1909 - val_loss: 1375.4448
    Epoch 171/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1274.0386 - val_loss: 1374.1985
    Epoch 172/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.4517 - val_loss: 1349.1561
    Epoch 173/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1274.0488 - val_loss: 1395.6627
    Epoch 174/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1272.6488 - val_loss: 1377.6416
    Epoch 175/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1269.7733 - val_loss: 1368.5010
    Epoch 176/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1268.2959 - val_loss: 1366.7430
    Epoch 177/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1275.9769 - val_loss: 1362.7006
    Epoch 178/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1276.7338 - val_loss: 1380.2163
    Epoch 179/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1269.8572 - val_loss: 1370.7141
    Epoch 180/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1274.5374 - val_loss: 1403.5394
    Epoch 181/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.8113 - val_loss: 1372.5513
    Epoch 182/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1276.8594 - val_loss: 1413.4908
    Epoch 183/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1269.3148 - val_loss: 1361.0903
    Epoch 184/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1268.8872 - val_loss: 1373.4523
    Epoch 185/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1271.5895 - val_loss: 1372.1611
    Epoch 186/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1270.2273 - val_loss: 1355.4773
    Epoch 187/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1272.4060 - val_loss: 1422.2743
    Epoch 188/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1268.8405 - val_loss: 1425.9403
    Epoch 189/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1270.7651 - val_loss: 1424.5466
    Epoch 190/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.2955 - val_loss: 1369.5966
    Epoch 191/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1269.1819 - val_loss: 1366.9432
    Epoch 192/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.5884 - val_loss: 1355.9956
    Epoch 193/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1270.4711 - val_loss: 1405.9330
    Epoch 194/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.8553 - val_loss: 1384.6250
    Epoch 195/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.9191 - val_loss: 1491.8761
    Epoch 196/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.0625 - val_loss: 1388.3187
    Epoch 197/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.6196 - val_loss: 1369.2097
    Epoch 198/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.4585 - val_loss: 1391.5360
    Epoch 199/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.6392 - val_loss: 1373.7816
    Epoch 200/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.6119 - val_loss: 1375.8647
    Epoch 201/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1270.4438 - val_loss: 1380.5945
    Epoch 202/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.8514 - val_loss: 1363.0153
    Epoch 203/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.4509 - val_loss: 1353.4664
    Epoch 204/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1269.2382 - val_loss: 1356.4694
    Epoch 205/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1263.8721 - val_loss: 1359.9911
    Epoch 206/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.6870 - val_loss: 1431.5447
    Epoch 207/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1268.4247 - val_loss: 1462.1410
    Epoch 208/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1272.1327 - val_loss: 1377.0243
    Epoch 209/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.1917 - val_loss: 1343.6891
    Epoch 210/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.1019 - val_loss: 1388.1656
    Epoch 211/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1268.5558 - val_loss: 1376.4495
    Epoch 212/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1265.9882 - val_loss: 1365.5338
    Epoch 213/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1268.4088 - val_loss: 1387.4215
    Epoch 214/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1263.8326 - val_loss: 1363.0562
    Epoch 215/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.8334 - val_loss: 1387.0813
    Epoch 216/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1265.4233 - val_loss: 1365.5372
    Epoch 217/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1265.4802 - val_loss: 1380.2494
    Epoch 218/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1262.6083 - val_loss: 1388.0400
    Epoch 219/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1265.7390 - val_loss: 1345.6804
    Epoch 220/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1264.2501 - val_loss: 1385.9811
    Epoch 221/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.4158 - val_loss: 1366.8619
    Epoch 222/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1264.4547 - val_loss: 1352.4104
    Epoch 223/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1264.1013 - val_loss: 1385.2520
    Epoch 224/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1263.7904 - val_loss: 1349.5062
    Epoch 225/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.7881 - val_loss: 1360.7559
    Epoch 226/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.1720 - val_loss: 1353.9856
    Epoch 227/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.6436 - val_loss: 1381.2483
    Epoch 228/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1264.5753 - val_loss: 1397.2802
    Epoch 229/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.1239 - val_loss: 1353.7999
    Epoch 230/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.7947 - val_loss: 1396.2168
    Epoch 231/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.3416 - val_loss: 1368.0713
    Epoch 232/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.4381 - val_loss: 1378.6190
    Epoch 233/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.6733 - val_loss: 1360.4879
    Epoch 234/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.2163 - val_loss: 1352.1178
    Epoch 235/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1266.7020 - val_loss: 1352.9268
    Epoch 236/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.7992 - val_loss: 1360.4727
    Epoch 237/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1264.0929 - val_loss: 1361.9320
    Epoch 238/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.9537 - val_loss: 1437.2678
    Epoch 239/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1263.5775 - val_loss: 1451.1812
    Epoch 240/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1262.9639 - val_loss: 1381.3680
    Epoch 241/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.9388 - val_loss: 1360.8447
    Epoch 242/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.8615 - val_loss: 1366.0323
    Epoch 243/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.1210 - val_loss: 1409.7771
    Epoch 244/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1262.8020 - val_loss: 1361.8848
    Epoch 245/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1262.3141 - val_loss: 1369.1321
    Epoch 246/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.6736 - val_loss: 1350.5479
    Epoch 247/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.0858 - val_loss: 1360.6149
    Epoch 248/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.8657 - val_loss: 1368.3392
    Epoch 249/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.2860 - val_loss: 1367.1464
    Epoch 250/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.8816 - val_loss: 1344.5361
    Epoch 251/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.2009 - val_loss: 1366.7781
    Epoch 252/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.5863 - val_loss: 1354.5193
    Epoch 253/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.5780 - val_loss: 1361.7860
    Epoch 254/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.1036 - val_loss: 1422.7787
    Epoch 255/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.9258 - val_loss: 1379.3344
    Epoch 256/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.1750 - val_loss: 1365.6913
    Epoch 257/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.1888 - val_loss: 1385.3246
    Epoch 258/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.7896 - val_loss: 1360.7689
    Epoch 259/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1255.4641 - val_loss: 1406.0463
    Epoch 260/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.9707 - val_loss: 1381.8934
    Epoch 261/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.8528 - val_loss: 1368.3265
    Epoch 262/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.4681 - val_loss: 1386.1698
    Epoch 263/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.5184 - val_loss: 1381.1787
    Epoch 264/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.8485 - val_loss: 1396.5172
    Epoch 265/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.1720 - val_loss: 1343.6761
    Epoch 266/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.4597 - val_loss: 1377.0321
    Epoch 267/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.9540 - val_loss: 1377.5350
    Epoch 268/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.6968 - val_loss: 1335.1879
    Epoch 269/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1263.9272 - val_loss: 1376.5044
    Epoch 270/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.0177 - val_loss: 1394.5807
    Epoch 271/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1302.8896 - val_loss: 1431.5850
    Epoch 272/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1286.7958 - val_loss: 1359.6661
    Epoch 273/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1273.1134 - val_loss: 1409.9794
    Epoch 274/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1276.1655 - val_loss: 1415.4042
    Epoch 275/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1272.4631 - val_loss: 1395.2605
    Epoch 276/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1270.2936 - val_loss: 1381.3314
    Epoch 277/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1262.3932 - val_loss: 1345.6893
    Epoch 278/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1267.1949 - val_loss: 1426.6649
    Epoch 279/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1262.7209 - val_loss: 1452.9142
    Epoch 280/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.5381 - val_loss: 1376.6248
    Epoch 281/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.6144 - val_loss: 1354.8582
    Epoch 282/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.4799 - val_loss: 1394.3427
    Epoch 283/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.0109 - val_loss: 1373.3654
    Epoch 284/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.0427 - val_loss: 1395.2216
    Epoch 285/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.3438 - val_loss: 1351.9698
    Epoch 286/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.4099 - val_loss: 1376.9637
    Epoch 287/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.5349 - val_loss: 1521.0823
    Epoch 288/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.6263 - val_loss: 1369.2642
    Epoch 289/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.8344 - val_loss: 1355.7256
    Epoch 290/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.1493 - val_loss: 1362.8881
    Epoch 291/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.0334 - val_loss: 1403.2531
    Epoch 292/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.6624 - val_loss: 1373.5095
    Epoch 293/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1255.0526 - val_loss: 1371.3037
    Epoch 294/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.3915 - val_loss: 1352.7286
    Epoch 295/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.5508 - val_loss: 1350.9298
    Epoch 296/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1259.4064 - val_loss: 1359.0381
    Epoch 297/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.2448 - val_loss: 1375.1986
    Epoch 298/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.9034 - val_loss: 1388.3687
    Epoch 299/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.5266 - val_loss: 1358.8616
    Epoch 300/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.8417 - val_loss: 1424.8640
    Epoch 301/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1255.6066 - val_loss: 1349.4690
    Epoch 302/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.9404 - val_loss: 1386.3380
    Epoch 303/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1261.5942 - val_loss: 1355.4481
    Epoch 304/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1255.7819 - val_loss: 1354.2739
    Epoch 305/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.5710 - val_loss: 1332.3513
    Epoch 306/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.5709 - val_loss: 1378.4596
    Epoch 307/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.0659 - val_loss: 1368.8263
    Epoch 308/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.2600 - val_loss: 1327.8877
    Epoch 309/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.7416 - val_loss: 1403.6859
    Epoch 310/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.3646 - val_loss: 1369.5760
    Epoch 311/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.4432 - val_loss: 1399.2726
    Epoch 312/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.4680 - val_loss: 1369.7795
    Epoch 313/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.7523 - val_loss: 1347.4818
    Epoch 314/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.8673 - val_loss: 1374.9731
    Epoch 315/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.8763 - val_loss: 1369.3168
    Epoch 316/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.0109 - val_loss: 1386.6633
    Epoch 317/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.6102 - val_loss: 1362.5634
    Epoch 318/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.9779 - val_loss: 1339.4950
    Epoch 319/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.4572 - val_loss: 1389.2463
    Epoch 320/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1263.1552 - val_loss: 1346.2550
    Epoch 321/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.5544 - val_loss: 1334.4785
    Epoch 322/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.4772 - val_loss: 1359.9871
    Epoch 323/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.3953 - val_loss: 1349.0449
    Epoch 324/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.3900 - val_loss: 1411.5540
    Epoch 325/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.9419 - val_loss: 1419.3018
    Epoch 326/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.8878 - val_loss: 1405.1940
    Epoch 327/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.2069 - val_loss: 1403.3402
    Epoch 328/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.6766 - val_loss: 1420.0620
    Epoch 329/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.5901 - val_loss: 1358.1097
    Epoch 330/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1255.9351 - val_loss: 1414.2864
    Epoch 331/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.9204 - val_loss: 1358.9930
    Epoch 332/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.2635 - val_loss: 1348.5930
    Epoch 333/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.5164 - val_loss: 1363.1550
    Epoch 334/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.6973 - val_loss: 1395.2815
    Epoch 335/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1258.0043 - val_loss: 1440.6101
    Epoch 336/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.2561 - val_loss: 1349.2465
    Epoch 337/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.2456 - val_loss: 1374.8879
    Epoch 338/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.4911 - val_loss: 1353.4170
    Epoch 339/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1255.9376 - val_loss: 1348.2977
    Epoch 340/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.3506 - val_loss: 1359.8834
    Epoch 341/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.4249 - val_loss: 1369.2833
    Epoch 342/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.1781 - val_loss: 1386.3599
    Epoch 343/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.3396 - val_loss: 1340.7831
    Epoch 344/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.8296 - val_loss: 1363.9120
    Epoch 345/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.6879 - val_loss: 1369.6038
    Epoch 346/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.5798 - val_loss: 1347.3872
    Epoch 347/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.0161 - val_loss: 1328.5443
    Epoch 348/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.3389 - val_loss: 1343.1138
    Epoch 349/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.0535 - val_loss: 1403.2299
    Epoch 350/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.1179 - val_loss: 1337.3441
    Epoch 351/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.7054 - val_loss: 1340.3584
    Epoch 352/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.5291 - val_loss: 1351.8914
    Epoch 353/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1253.8191 - val_loss: 1358.7744
    Epoch 354/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.3726 - val_loss: 1348.4819
    Epoch 355/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.3142 - val_loss: 1333.0215
    Epoch 356/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.5247 - val_loss: 1368.0166
    Epoch 357/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.3497 - val_loss: 1428.1327
    Epoch 358/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.6400 - val_loss: 1363.9940
    Epoch 359/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.9768 - val_loss: 1367.2742
    Epoch 360/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.3049 - val_loss: 1354.2850
    Epoch 361/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.3263 - val_loss: 1338.1881
    Epoch 362/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.8883 - val_loss: 1342.7039
    Epoch 363/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.6216 - val_loss: 1351.3154
    Epoch 364/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.8972 - val_loss: 1354.4270
    Epoch 365/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.0312 - val_loss: 1341.5944
    Epoch 366/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.2833 - val_loss: 1341.1285
    Epoch 367/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.7550 - val_loss: 1358.7091
    Epoch 368/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.2279 - val_loss: 1357.4357
    Epoch 369/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.7603 - val_loss: 1374.3285
    Epoch 370/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.3293 - val_loss: 1336.9712
    Epoch 371/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.3035 - val_loss: 1331.5042
    Epoch 372/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.4591 - val_loss: 1325.9847
    Epoch 373/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.0770 - val_loss: 1375.1504
    Epoch 374/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.7225 - val_loss: 1382.1328
    Epoch 375/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.0936 - val_loss: 1376.3654
    Epoch 376/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1243.4028 - val_loss: 1361.2617
    Epoch 377/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.8911 - val_loss: 1374.8837
    Epoch 378/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.1824 - val_loss: 1358.4695
    Epoch 379/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.9814 - val_loss: 1334.0447
    Epoch 380/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.9514 - val_loss: 1360.7159
    Epoch 381/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.5676 - val_loss: 1382.8361
    Epoch 382/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.9301 - val_loss: 1379.1356
    Epoch 383/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.8286 - val_loss: 1357.8649
    Epoch 384/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1246.5825 - val_loss: 1353.3876
    Epoch 385/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.4185 - val_loss: 1361.0260
    Epoch 386/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1255.1479 - val_loss: 1360.8904
    Epoch 387/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.4341 - val_loss: 1361.7378
    Epoch 388/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.0577 - val_loss: 1339.6954
    Epoch 389/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.9921 - val_loss: 1412.5791
    Epoch 390/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.9587 - val_loss: 1356.9666
    Epoch 391/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.2981 - val_loss: 1423.0720
    Epoch 392/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.0753 - val_loss: 1404.0002
    Epoch 393/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.9835 - val_loss: 1356.3676
    Epoch 394/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.1608 - val_loss: 1394.3966
    Epoch 395/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.1560 - val_loss: 1349.7224
    Epoch 396/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1240.8638 - val_loss: 1380.8447
    Epoch 397/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1243.3535 - val_loss: 1383.9667
    Epoch 398/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.0054 - val_loss: 1410.9551
    Epoch 399/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.1686 - val_loss: 1328.9204
    Epoch 400/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.5348 - val_loss: 1342.9916
    Epoch 401/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.7169 - val_loss: 1397.9393
    Epoch 402/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.9663 - val_loss: 1368.0870
    Epoch 403/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.0129 - val_loss: 1355.0293
    Epoch 404/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.6324 - val_loss: 1425.2419
    Epoch 405/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.2440 - val_loss: 1400.2473
    Epoch 406/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.1926 - val_loss: 1337.7585
    Epoch 407/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.8425 - val_loss: 1392.8839
    Epoch 408/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.7399 - val_loss: 1335.1840
    Epoch 409/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.7710 - val_loss: 1355.1683
    Epoch 410/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1243.2037 - val_loss: 1335.2997
    Epoch 411/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.8079 - val_loss: 1346.3903
    Epoch 412/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.2179 - val_loss: 1342.8040
    Epoch 413/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.6479 - val_loss: 1359.8762
    Epoch 414/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.8409 - val_loss: 1342.9470
    Epoch 415/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.5378 - val_loss: 1382.4384
    Epoch 416/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.5505 - val_loss: 1350.1709
    Epoch 417/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.7450 - val_loss: 1357.3219
    Epoch 418/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.8068 - val_loss: 1325.0756
    Epoch 419/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.9132 - val_loss: 1382.1514
    Epoch 420/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.4307 - val_loss: 1325.9026
    Epoch 421/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.7330 - val_loss: 1401.6067
    Epoch 422/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1250.3646 - val_loss: 1364.2948
    Epoch 423/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.5105 - val_loss: 1349.3448
    Epoch 424/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.6586 - val_loss: 1398.8160
    Epoch 425/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.8146 - val_loss: 1382.7811
    Epoch 426/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.8740 - val_loss: 1349.3864
    Epoch 427/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.7977 - val_loss: 1385.1957
    Epoch 428/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.2921 - val_loss: 1343.2153
    Epoch 429/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.6100 - val_loss: 1353.8077
    Epoch 430/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.9506 - val_loss: 1358.2061
    Epoch 431/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.3729 - val_loss: 1349.8113
    Epoch 432/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1245.9709 - val_loss: 1365.4252
    Epoch 433/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.3986 - val_loss: 1374.9568
    Epoch 434/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.3701 - val_loss: 1332.0798
    Epoch 435/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.5295 - val_loss: 1341.4617
    Epoch 436/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.4125 - val_loss: 1347.1903
    Epoch 437/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1240.1387 - val_loss: 1334.2456
    Epoch 438/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.4988 - val_loss: 1402.3507
    Epoch 439/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.8900 - val_loss: 1358.6298
    Epoch 440/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.1818 - val_loss: 1365.0961
    Epoch 441/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.2858 - val_loss: 1335.9127
    Epoch 442/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1243.5807 - val_loss: 1341.7614
    Epoch 443/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.4757 - val_loss: 1345.5536
    Epoch 444/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.0743 - val_loss: 1403.9386
    Epoch 445/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1251.6803 - val_loss: 1343.5829
    Epoch 446/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.2145 - val_loss: 1369.7220
    Epoch 447/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.3521 - val_loss: 1386.9506
    Epoch 448/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1254.5229 - val_loss: 1354.3682
    Epoch 449/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1240.8041 - val_loss: 1343.2454
    Epoch 450/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.1990 - val_loss: 1329.7723
    Epoch 451/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.8624 - val_loss: 1354.2744
    Epoch 452/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.0172 - val_loss: 1349.0110
    Epoch 453/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.9260 - val_loss: 1349.2555
    Epoch 454/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.5671 - val_loss: 1354.8837
    Epoch 455/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.6656 - val_loss: 1371.1841
    Epoch 456/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.6715 - val_loss: 1352.6292
    Epoch 457/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.4617 - val_loss: 1333.1060
    Epoch 458/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.7552 - val_loss: 1311.3289
    Epoch 459/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.2393 - val_loss: 1376.5481
    Epoch 460/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.9889 - val_loss: 1335.4241
    Epoch 461/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.2288 - val_loss: 1368.5680
    Epoch 462/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1243.7437 - val_loss: 1352.3932
    Epoch 463/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.9602 - val_loss: 1380.7339
    Epoch 464/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.2715 - val_loss: 1361.1790
    Epoch 465/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.9427 - val_loss: 1351.8770
    Epoch 466/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.2408 - val_loss: 1366.7878
    Epoch 467/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.5453 - val_loss: 1331.6241
    Epoch 468/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.9485 - val_loss: 1320.0457
    Epoch 469/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.8628 - val_loss: 1339.9023
    Epoch 470/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.6232 - val_loss: 1395.3525
    Epoch 471/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.4279 - val_loss: 1323.4525
    Epoch 472/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.0702 - val_loss: 1348.0216
    Epoch 473/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.6373 - val_loss: 1339.9539
    Epoch 474/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1247.2129 - val_loss: 1366.1527
    Epoch 475/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.5942 - val_loss: 1351.1713
    Epoch 476/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.8477 - val_loss: 1390.8986
    Epoch 477/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.9015 - val_loss: 1443.6365
    Epoch 478/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.9291 - val_loss: 1333.4618
    Epoch 479/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.2527 - val_loss: 1405.0822
    Epoch 480/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.6240 - val_loss: 1355.3951
    Epoch 481/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1240.7397 - val_loss: 1473.4386
    Epoch 482/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.2921 - val_loss: 1324.0913
    Epoch 483/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.0298 - val_loss: 1364.9191
    Epoch 484/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.9773 - val_loss: 1366.4880
    Epoch 485/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1249.4779 - val_loss: 1399.3457
    Epoch 486/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.7458 - val_loss: 1335.6949
    Epoch 487/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1234.7476 - val_loss: 1365.3229
    Epoch 488/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.7416 - val_loss: 1322.8060
    Epoch 489/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.8752 - val_loss: 1340.6396
    Epoch 490/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1236.6908 - val_loss: 1346.8629
    Epoch 491/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.3922 - val_loss: 1335.6091
    Epoch 492/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.6654 - val_loss: 1324.8318
    Epoch 493/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.4746 - val_loss: 1351.8192
    Epoch 494/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.4713 - val_loss: 1355.8499
    Epoch 495/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1240.6869 - val_loss: 1347.9711
    Epoch 496/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.9443 - val_loss: 1320.8361
    Epoch 497/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.9917 - val_loss: 1366.6306
    Epoch 498/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.9143 - val_loss: 1353.7350
    Epoch 499/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1243.6449 - val_loss: 1376.1250
    Epoch 500/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.4991 - val_loss: 1350.6459
    Epoch 501/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1236.9333 - val_loss: 1382.6921
    Epoch 502/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1243.1056 - val_loss: 1366.9171
    Epoch 503/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.9376 - val_loss: 1343.6237
    Epoch 504/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.3488 - val_loss: 1351.9529
    Epoch 505/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1240.5577 - val_loss: 1353.7401
    Epoch 506/1000
    2424/2424 [==============================] - 4s 1ms/step - loss: 1246.7107 - val_loss: 1345.6600
    Epoch 507/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1244.9648 - val_loss: 1346.0579
    Epoch 508/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1256.7574 - val_loss: 1357.0903
    Epoch 509/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1252.3302 - val_loss: 1379.8123
    Epoch 510/1000
    2424/2424 [==============================] - 4s 2ms/step - loss: 1250.4573 - val_loss: 1364.3579
    Epoch 511/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.1187 - val_loss: 1355.0359
    Epoch 512/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.7356 - val_loss: 1328.0511
    Epoch 513/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1235.8590 - val_loss: 1373.3347
    Epoch 514/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1231.0608 - val_loss: 1380.6416
    Epoch 515/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1234.1034 - val_loss: 1379.2932
    Epoch 516/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.7863 - val_loss: 1403.0165
    Epoch 517/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1235.5370 - val_loss: 1434.4629
    Epoch 518/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.6022 - val_loss: 1322.6416
    Epoch 519/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1236.1560 - val_loss: 1361.2306
    Epoch 520/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.9530 - val_loss: 1336.1486
    Epoch 521/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1234.6815 - val_loss: 1371.8948
    Epoch 522/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1227.6832 - val_loss: 1350.6053
    Epoch 523/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.5842 - val_loss: 1342.9209
    Epoch 524/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1246.6826 - val_loss: 1346.5826
    Epoch 525/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.9989 - val_loss: 1366.3342
    Epoch 526/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.8102 - val_loss: 1338.8673
    Epoch 527/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1240.1780 - val_loss: 1354.1885
    Epoch 528/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.4017 - val_loss: 1359.9838
    Epoch 529/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1235.6522 - val_loss: 1347.9667
    Epoch 530/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1248.9448 - val_loss: 1358.5205
    Epoch 531/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.6427 - val_loss: 1343.0199
    Epoch 532/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.8402 - val_loss: 1402.7451
    Epoch 533/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.6689 - val_loss: 1407.5715
    Epoch 534/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.3033 - val_loss: 1472.0634
    Epoch 535/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.5170 - val_loss: 1368.9307
    Epoch 536/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1232.9141 - val_loss: 1336.1913
    Epoch 537/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1257.7297 - val_loss: 1329.9816
    Epoch 538/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1236.9838 - val_loss: 1352.7369
    Epoch 539/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1231.9520 - val_loss: 1335.0166
    Epoch 540/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.2416 - val_loss: 1316.7045
    Epoch 541/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.6423 - val_loss: 1384.4249
    Epoch 542/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1241.3521 - val_loss: 1333.0133
    Epoch 543/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1234.5223 - val_loss: 1353.9980
    Epoch 544/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1231.3785 - val_loss: 1366.1263
    Epoch 545/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1236.9264 - val_loss: 1341.4358
    Epoch 546/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1235.1544 - val_loss: 1331.9294
    Epoch 547/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1236.4382 - val_loss: 1340.4929
    Epoch 548/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.5925 - val_loss: 1335.0614
    Epoch 549/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1237.1338 - val_loss: 1341.9058
    Epoch 550/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.0481 - val_loss: 1408.6301
    Epoch 551/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1238.6630 - val_loss: 1336.0790
    Epoch 552/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1229.1145 - val_loss: 1375.6992
    Epoch 553/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1235.7731 - val_loss: 1367.4312
    Epoch 554/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1236.8540 - val_loss: 1368.1880
    Epoch 555/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1239.3013 - val_loss: 1339.0579
    Epoch 556/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1242.6877 - val_loss: 1332.1714
    Epoch 557/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1260.6245 - val_loss: 1501.7150
    Epoch 558/1000
    2424/2424 [==============================] - 3s 1ms/step - loss: 1245.4034 - val_loss: 1347.1127


The root mean squared errors for training and testing set are shown below:


```python
# Traing RMSE
np.sqrt(mean_squared_error(y_train,best_nn.predict(X_train)))
```

    2424/2424 [==============================] - 2s 597us/step





    34.78459569413211




```python
# Testing RMSE
np.sqrt(mean_squared_error(y_test,best_nn.predict(X_test)))
```

    485/485 [==============================] - 0s 612us/step





    34.821059741172114



The learning curve is shown below.


```python
plt.plot(history_best_nn.history['loss'],label = "loss")
plt.plot(history_best_nn.history['val_loss'], label = "val_loss")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epoch")
```




    Text(0.5, 0, 'Epoch')





![png](/assets/output_157_1.png)



### Convolutional Neural Network (CNN)

The structure of the convolutional network is shown below.


```python
model_conv = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32,2,activation = 'relu',input_shape = (14,1,), kernel_initializer="he_normal",padding = 'same'),
    tf.keras.layers.Conv1D(32,3,activation = 'relu', kernel_initializer="he_normal",padding = 'same'),
    tf.keras.layers.Conv1D(32,3,activation = 'relu', kernel_initializer="he_normal"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dense(64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dense(32, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dense(1, activation= None)
])


```


```python
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,restore_best_weights=True)
model_conv.compile(loss="mse", optimizer="nadam")
history_cov = model_conv.fit(X_train, y_train, epochs=1000,
                    validation_data=(X_valid, y_valid),callbacks=[callback])
```

    Epoch 1/1000
    2424/2424 [==============================] - 7s 2ms/step - loss: 1632.2096 - val_loss: 1460.4231
    Epoch 2/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1373.3662 - val_loss: 1505.1824
    Epoch 3/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1366.6722 - val_loss: 1474.0969
    Epoch 4/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1346.9469 - val_loss: 1475.6388
    Epoch 5/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1334.1947 - val_loss: 1547.0897
    Epoch 6/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1323.7288 - val_loss: 1439.5723
    Epoch 7/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1310.3964 - val_loss: 1432.5355
    Epoch 8/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1294.4939 - val_loss: 1449.5792
    Epoch 9/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1290.2340 - val_loss: 1363.4564
    Epoch 10/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1283.3922 - val_loss: 1426.5688
    Epoch 11/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1278.2701 - val_loss: 1379.9060
    Epoch 12/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1275.8239 - val_loss: 1353.9603
    Epoch 13/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1267.0742 - val_loss: 1378.1696
    Epoch 14/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1264.0004 - val_loss: 1390.1466
    Epoch 15/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1260.7502 - val_loss: 1392.5708
    Epoch 16/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1254.7893 - val_loss: 1356.3167
    Epoch 17/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1255.6683 - val_loss: 1362.1136
    Epoch 18/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1245.3850 - val_loss: 1397.6771
    Epoch 19/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1244.1243 - val_loss: 1356.0706
    Epoch 20/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1237.3307 - val_loss: 1335.0283
    Epoch 21/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1235.4623 - val_loss: 1332.3763
    Epoch 22/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1230.7083 - val_loss: 1322.7377
    Epoch 23/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1231.0759 - val_loss: 1325.8740
    Epoch 24/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1227.9644 - val_loss: 1315.2830
    Epoch 25/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1226.1245 - val_loss: 1331.6392
    Epoch 26/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1225.0741 - val_loss: 1311.2473
    Epoch 27/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1221.7261 - val_loss: 1300.4327
    Epoch 28/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1217.8020 - val_loss: 1314.4773
    Epoch 29/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1218.2078 - val_loss: 1333.0670
    Epoch 30/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1215.9946 - val_loss: 1278.1498
    Epoch 31/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1210.2246 - val_loss: 1308.6256
    Epoch 32/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1211.1191 - val_loss: 1282.7308
    Epoch 33/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1209.0834 - val_loss: 1282.1754
    Epoch 34/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1222.3536 - val_loss: 1284.7974
    Epoch 35/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1210.3438 - val_loss: 1307.2913
    Epoch 36/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1209.0089 - val_loss: 1324.7301
    Epoch 37/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1208.5164 - val_loss: 1354.2831
    Epoch 38/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1204.0726 - val_loss: 1319.6295
    Epoch 39/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1203.0276 - val_loss: 1294.0881
    Epoch 40/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1201.9962 - val_loss: 1280.3878
    Epoch 41/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1199.2607 - val_loss: 1295.9388
    Epoch 42/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1202.5725 - val_loss: 1255.4443
    Epoch 43/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1197.9132 - val_loss: 1253.5717
    Epoch 44/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1195.5591 - val_loss: 1253.0435
    Epoch 45/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1194.6122 - val_loss: 1311.2078
    Epoch 46/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1192.3362 - val_loss: 1263.1780
    Epoch 47/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1189.3467 - val_loss: 1267.4291
    Epoch 48/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1187.6294 - val_loss: 1271.2804
    Epoch 49/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1189.1650 - val_loss: 1240.1514
    Epoch 50/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1183.8344 - val_loss: 1244.5862
    Epoch 51/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1190.6515 - val_loss: 1255.4740
    Epoch 52/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1181.4845 - val_loss: 1289.4598
    Epoch 53/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1182.6373 - val_loss: 1258.4595
    Epoch 54/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1180.4437 - val_loss: 1275.0465
    Epoch 55/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1181.1841 - val_loss: 1269.8691
    Epoch 56/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1177.5139 - val_loss: 1223.5607
    Epoch 57/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1173.1550 - val_loss: 1232.9540
    Epoch 58/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1175.2343 - val_loss: 1249.2113
    Epoch 59/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1173.5836 - val_loss: 1236.4746
    Epoch 60/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1172.7565 - val_loss: 1278.0226
    Epoch 61/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1170.5190 - val_loss: 1217.3357
    Epoch 62/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1170.4573 - val_loss: 1220.4858
    Epoch 63/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1173.0367 - val_loss: 1229.4474
    Epoch 64/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1167.6458 - val_loss: 1260.2000
    Epoch 65/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1167.0093 - val_loss: 1249.1899
    Epoch 66/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1165.3370 - val_loss: 1267.4150
    Epoch 67/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1171.3475 - val_loss: 1257.6511
    Epoch 68/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1162.6521 - val_loss: 1238.8892
    Epoch 69/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1164.6411 - val_loss: 1246.6920
    Epoch 70/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1162.0336 - val_loss: 1317.0363
    Epoch 71/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1163.0983 - val_loss: 1217.4580
    Epoch 72/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1159.1715 - val_loss: 1229.0195
    Epoch 73/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1155.4551 - val_loss: 1232.8341
    Epoch 74/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1155.5284 - val_loss: 1273.4977
    Epoch 75/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1150.9237 - val_loss: 1256.0906
    Epoch 76/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1158.6924 - val_loss: 1219.9215
    Epoch 77/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1151.5303 - val_loss: 1210.6943
    Epoch 78/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1156.8185 - val_loss: 1243.4762
    Epoch 79/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1149.7745 - val_loss: 1240.2955
    Epoch 80/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1148.9011 - val_loss: 1251.2832
    Epoch 81/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1148.1597 - val_loss: 1188.1097
    Epoch 82/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1145.2150 - val_loss: 1197.4039
    Epoch 83/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1139.6521 - val_loss: 1259.9893
    Epoch 84/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1146.0648 - val_loss: 1288.2117
    Epoch 85/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1143.6884 - val_loss: 1251.1246
    Epoch 86/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1137.1627 - val_loss: 1304.6818
    Epoch 87/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1142.2595 - val_loss: 1216.3152
    Epoch 88/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1139.2599 - val_loss: 1209.3552
    Epoch 89/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1144.9933 - val_loss: 1215.9009
    Epoch 90/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1134.8512 - val_loss: 1226.0697
    Epoch 91/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1139.0433 - val_loss: 1191.5439
    Epoch 92/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1135.2897 - val_loss: 1233.7103
    Epoch 93/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1133.8296 - val_loss: 1185.4534
    Epoch 94/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1131.6901 - val_loss: 1220.9338
    Epoch 95/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1131.1788 - val_loss: 1192.0964
    Epoch 96/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1130.4845 - val_loss: 1188.5325
    Epoch 97/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1129.6383 - val_loss: 1205.8627
    Epoch 98/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1121.6267 - val_loss: 1208.7793
    Epoch 99/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1131.1461 - val_loss: 1211.6389
    Epoch 100/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1128.0181 - val_loss: 1218.3246
    Epoch 101/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1121.8365 - val_loss: 1258.9097
    Epoch 102/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1123.6489 - val_loss: 1200.2845
    Epoch 103/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1122.6870 - val_loss: 1192.6638
    Epoch 104/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1119.6632 - val_loss: 1201.2687
    Epoch 105/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1120.4332 - val_loss: 1237.7335
    Epoch 106/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1124.2760 - val_loss: 1206.6163
    Epoch 107/1000
    2424/2424 [==============================] - 6s 3ms/step - loss: 1115.9065 - val_loss: 1191.7129
    Epoch 108/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1114.9535 - val_loss: 1220.4253
    Epoch 109/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1113.9335 - val_loss: 1169.8805
    Epoch 110/1000
    2424/2424 [==============================] - 6s 3ms/step - loss: 1114.1995 - val_loss: 1203.9440
    Epoch 111/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1114.7330 - val_loss: 1179.9154
    Epoch 112/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1110.5181 - val_loss: 1226.2498
    Epoch 113/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1111.8418 - val_loss: 1175.3497
    Epoch 114/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1107.7721 - val_loss: 1243.4736
    Epoch 115/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1107.0792 - val_loss: 1204.7606
    Epoch 116/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1106.8302 - val_loss: 1218.7834
    Epoch 117/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1103.2577 - val_loss: 1193.5872
    Epoch 118/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1105.6810 - val_loss: 1204.3000
    Epoch 119/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1108.2013 - val_loss: 1198.3353
    Epoch 120/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1100.8220 - val_loss: 1171.2262
    Epoch 121/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1098.3665 - val_loss: 1221.9154
    Epoch 122/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1089.1938 - val_loss: 1222.1921
    Epoch 123/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1101.3915 - val_loss: 1216.4297
    Epoch 124/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1098.6339 - val_loss: 1194.1328
    Epoch 125/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1096.5940 - val_loss: 1202.3700
    Epoch 126/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1096.3538 - val_loss: 1171.2760
    Epoch 127/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1090.3364 - val_loss: 1226.1420
    Epoch 128/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1083.3464 - val_loss: 1189.9419
    Epoch 129/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1100.6931 - val_loss: 1225.2375
    Epoch 130/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1089.3152 - val_loss: 1205.6100
    Epoch 131/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1092.3683 - val_loss: 1201.2443
    Epoch 132/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1091.9420 - val_loss: 1198.7275
    Epoch 133/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1085.7850 - val_loss: 1177.8109
    Epoch 134/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1090.7443 - val_loss: 1217.5830
    Epoch 135/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1080.8724 - val_loss: 1185.1119
    Epoch 136/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1083.1538 - val_loss: 1171.8986
    Epoch 137/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1084.3926 - val_loss: 1153.3682
    Epoch 138/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1083.0117 - val_loss: 1196.5400
    Epoch 139/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1077.8282 - val_loss: 1173.7222
    Epoch 140/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1081.1149 - val_loss: 1192.7034
    Epoch 141/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1079.6298 - val_loss: 1268.9935
    Epoch 142/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1080.1732 - val_loss: 1283.7703
    Epoch 143/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1089.0325 - val_loss: 1149.7229
    Epoch 144/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1074.5051 - val_loss: 1183.3925
    Epoch 145/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1074.3197 - val_loss: 1177.6888
    Epoch 146/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1076.5251 - val_loss: 1184.1573
    Epoch 147/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1077.1130 - val_loss: 1137.1984
    Epoch 148/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1069.9033 - val_loss: 1154.5635
    Epoch 149/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1066.9619 - val_loss: 1166.2358
    Epoch 150/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1067.5448 - val_loss: 1196.7928
    Epoch 151/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1069.5793 - val_loss: 1136.7689
    Epoch 152/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1067.5372 - val_loss: 1163.7437
    Epoch 153/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1058.6735 - val_loss: 1220.3914
    Epoch 154/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1070.7375 - val_loss: 1209.5841
    Epoch 155/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1069.6257 - val_loss: 1176.1388
    Epoch 156/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1078.7805 - val_loss: 1222.3766
    Epoch 157/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1112.8574 - val_loss: 1222.7164
    Epoch 158/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1065.3074 - val_loss: 1144.2107
    Epoch 159/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1061.4540 - val_loss: 1156.4764
    Epoch 160/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1059.5453 - val_loss: 1201.5692
    Epoch 161/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1061.7609 - val_loss: 1194.7961
    Epoch 162/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1055.1875 - val_loss: 1151.3364
    Epoch 163/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1064.2267 - val_loss: 1170.3516
    Epoch 164/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1059.2406 - val_loss: 1132.3406
    Epoch 165/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1063.6415 - val_loss: 1225.3683
    Epoch 166/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1060.6193 - val_loss: 1134.6448
    Epoch 167/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1057.7936 - val_loss: 1182.0817
    Epoch 168/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1056.3209 - val_loss: 1225.5228
    Epoch 169/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1064.7632 - val_loss: 1161.3274
    Epoch 170/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1056.5997 - val_loss: 1160.1941
    Epoch 171/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1053.7512 - val_loss: 1160.8353
    Epoch 172/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1054.1938 - val_loss: 1146.2437
    Epoch 173/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1054.0056 - val_loss: 1183.4795
    Epoch 174/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1053.1650 - val_loss: 1211.8622
    Epoch 175/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1052.6007 - val_loss: 1196.0558
    Epoch 176/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1042.5089 - val_loss: 1116.6172
    Epoch 177/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1045.4923 - val_loss: 1185.9246
    Epoch 178/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1049.2926 - val_loss: 1152.5020
    Epoch 179/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1054.4542 - val_loss: 1242.9181
    Epoch 180/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1049.8531 - val_loss: 1184.9471
    Epoch 181/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1040.7905 - val_loss: 1131.3975
    Epoch 182/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1049.2712 - val_loss: 1154.3792
    Epoch 183/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1035.3655 - val_loss: 1120.8547
    Epoch 184/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1044.4467 - val_loss: 1161.6073
    Epoch 185/1000
    2424/2424 [==============================] - 7s 3ms/step - loss: 1046.8494 - val_loss: 1147.8804
    Epoch 186/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1047.1658 - val_loss: 1189.9502
    Epoch 187/1000
    2424/2424 [==============================] - 6s 2ms/step - loss: 1039.9266 - val_loss: 1147.5398
    Epoch 188/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1042.9817 - val_loss: 1196.4209
    Epoch 189/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1039.0784 - val_loss: 1137.0851
    Epoch 190/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1038.5880 - val_loss: 1157.4337
    Epoch 191/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1040.5125 - val_loss: 1181.9531
    Epoch 192/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1041.4628 - val_loss: 1144.9553
    Epoch 193/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1043.2192 - val_loss: 1126.1615
    Epoch 194/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1037.1057 - val_loss: 1140.5917
    Epoch 195/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1041.4860 - val_loss: 1109.2225
    Epoch 196/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1035.4133 - val_loss: 1167.8536
    Epoch 197/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1039.4752 - val_loss: 1136.7598
    Epoch 198/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1027.4529 - val_loss: 1193.4949
    Epoch 199/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1031.9694 - val_loss: 1165.2278
    Epoch 200/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1031.4280 - val_loss: 1148.3068
    Epoch 201/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1027.9086 - val_loss: 1210.2145
    Epoch 202/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1029.5967 - val_loss: 1194.2570
    Epoch 203/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1036.6545 - val_loss: 1209.9480
    Epoch 204/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1027.7101 - val_loss: 1142.8517
    Epoch 205/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1026.1581 - val_loss: 1176.1245
    Epoch 206/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1036.5669 - val_loss: 1169.6749
    Epoch 207/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1023.5724 - val_loss: 1151.7473
    Epoch 208/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1029.5367 - val_loss: 1233.7201
    Epoch 209/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1028.0232 - val_loss: 1169.1259
    Epoch 210/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1026.2405 - val_loss: 1143.0227
    Epoch 211/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1029.3799 - val_loss: 1194.1953
    Epoch 212/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1024.0352 - val_loss: 1196.0070
    Epoch 213/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1029.2903 - val_loss: 1182.8269
    Epoch 214/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1030.3057 - val_loss: 1140.3273
    Epoch 215/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1029.1404 - val_loss: 1114.3191
    Epoch 216/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1021.1459 - val_loss: 1136.9663
    Epoch 217/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1020.5687 - val_loss: 1136.4908
    Epoch 218/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1023.6076 - val_loss: 1168.2968
    Epoch 219/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1021.1273 - val_loss: 1172.9546
    Epoch 220/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1020.8776 - val_loss: 1110.9832
    Epoch 221/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1015.9850 - val_loss: 1246.4061
    Epoch 222/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1024.9508 - val_loss: 1166.5314
    Epoch 223/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1021.7601 - val_loss: 1132.3287
    Epoch 224/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1022.7983 - val_loss: 1165.6740
    Epoch 225/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1018.3124 - val_loss: 1170.6283
    Epoch 226/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1017.9795 - val_loss: 1195.0621
    Epoch 227/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1018.9117 - val_loss: 1195.9948
    Epoch 228/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1011.5366 - val_loss: 1160.3621
    Epoch 229/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1020.4540 - val_loss: 1142.8944
    Epoch 230/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1014.4623 - val_loss: 1238.8601
    Epoch 231/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1016.9763 - val_loss: 1203.5620
    Epoch 232/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1008.4352 - val_loss: 1152.4442
    Epoch 233/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1016.5429 - val_loss: 1123.7742
    Epoch 234/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1009.5654 - val_loss: 1160.1962
    Epoch 235/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1010.1576 - val_loss: 1142.0682
    Epoch 236/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1011.2394 - val_loss: 1164.1600
    Epoch 237/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1047.6915 - val_loss: 1183.9366
    Epoch 238/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1014.8844 - val_loss: 1119.3680
    Epoch 239/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1003.3881 - val_loss: 1250.4004
    Epoch 240/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1018.6555 - val_loss: 1147.4958
    Epoch 241/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1007.6484 - val_loss: 1173.2128
    Epoch 242/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1007.8110 - val_loss: 1131.7792
    Epoch 243/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1003.6499 - val_loss: 1127.2411
    Epoch 244/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1013.4796 - val_loss: 1174.7893
    Epoch 245/1000
    2424/2424 [==============================] - 5s 2ms/step - loss: 1002.9050 - val_loss: 1177.7996



```python
model_conv.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv1d (Conv1D)             (None, 14, 32)            96        

     conv1d_1 (Conv1D)           (None, 14, 32)            3104      

     conv1d_2 (Conv1D)           (None, 12, 32)            3104      

     flatten (Flatten)           (None, 384)               0         

     dense_8 (Dense)             (None, 128)               49280     

     dense_9 (Dense)             (None, 64)                8256      

     dense_10 (Dense)            (None, 32)                2080      

     dense_11 (Dense)            (None, 1)                 33        

    =================================================================
    Total params: 65,953
    Trainable params: 65,953
    Non-trainable params: 0
    _________________________________________________________________


The learning curve is shown below.


```python
plt.plot(history_cov.history['loss'],label = "loss")
plt.plot(history_cov.history['val_loss'], label = "val_loss")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Epoch")
```




    Text(0.5, 0, 'Epoch')





![png](/assets/output_164_1.png)



The root mean squared errors for training and testing set are shown below:


```python
# Training RMSE
np.sqrt(mean_squared_error(y_train,model_conv.predict(X_train)))
```

    2424/2424 [==============================] - 2s 920us/step





    31.66131974960281




```python
# Testing RMSE
np.sqrt(mean_squared_error(y_test,model_conv.predict(X_test)))
```

    485/485 [==============================] - 0s 814us/step





    32.372653521773714



## Result

The Root Mean Square Error for each machine learning models are shown below, together with the chosen hyperparamters.


```python
res = pd.DataFrame({"Model": ["Binomial Tree",'Linear Regression',"Polynomial Regression","Ridge Regression","Lasso Regression","Linear SVR","Nonlinear SVR","Random Forest","KNN","Neural Network", "Neural Network (tuned)", " Convolutional Neutral Network"],
                   "RMSE(training)":[None,
38.43095625260875,
36.17182222027887,
38.43095969174494,
38.431561232353104,
38.547851070665025,
37.808085641917536,
31.349066580782836,
31.92967572127278,
36.21230441027238,
34.78459569413211,
31.66131974960281
],
                "RMSE(testing)":
                   [48.219430763590715,
                    37.97049534873681,
36.06896946882215,
37.97019255417378,
37.969146928057036,
38.03050061254464,
37.92867379033997,
32.90546500222834,
32.912277382176526,
35.71456726735933,
34.821059741172114,
32.372653521773714
],
"Hyperparameters" :
        ["29 steps","",
"degree = 2",
"alpha = 10",
"alpha = 0.01",
"C=10, epsilon=10",
"C=100, epsilon=1",
"ccp_alpha=0, max_depth=8, n_estimators=400, bootstrap = False",
         "neighbors = 15",
        "consider model summary above",
        "consider model summary above",
        "consider model summary above"]    
                   })
res
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE(training)</th>
      <th>RMSE(testing)</th>
      <th>Hyperparameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Binomial Tree</td>
      <td>NaN</td>
      <td>48.219431</td>
      <td>29 steps</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear Regression</td>
      <td>38.430956</td>
      <td>37.970495</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Polynomial Regression</td>
      <td>36.171822</td>
      <td>36.068969</td>
      <td>degree = 2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ridge Regression</td>
      <td>38.430960</td>
      <td>37.970193</td>
      <td>alpha = 10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lasso Regression</td>
      <td>38.431561</td>
      <td>37.969147</td>
      <td>alpha = 0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Linear SVR</td>
      <td>38.547851</td>
      <td>38.030501</td>
      <td>C=10, epsilon=10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Nonlinear SVR</td>
      <td>37.808086</td>
      <td>37.928674</td>
      <td>C=100, epsilon=1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Random Forest</td>
      <td>31.349067</td>
      <td>32.905465</td>
      <td>ccp_alpha=0, max_depth=8, n_estimators=400, bo...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>KNN</td>
      <td>31.929676</td>
      <td>32.912277</td>
      <td>neighbors = 25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Neural Network</td>
      <td>36.212304</td>
      <td>35.714567</td>
      <td>consider model summary above</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Neural Network (tuned)</td>
      <td>34.784596</td>
      <td>34.821060</td>
      <td>consider model summary above</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Convolutional Neutral Network</td>
      <td>31.661320</td>
      <td>32.372654</td>
      <td>consider model summary above</td>
    </tr>
  </tbody>
</table>
</div>



Consider the RMSE for the testing set, the lower RMSE, the more accurate the model to compute option price. By this sole metric, all the machine learning model we used in this project performs better than the binomial tree.
By ranking the models by this value, we have that the Convolutional Neural Network performs the best, the random forest is the second best, and the K-Nearest Neighbors is the third.

The statistics for the absolute error for the best three models and the binomial tree are shown below.


```python
res_error = pd.concat([pd.DataFrame({"binomial abs_error": np.abs(np.array(bm_list) -y_test.to_numpy().flatten())}).describe(),
                        np.abs(model_conv.predict(X_test).ravel() - y_test).describe(),
          np.abs((best_rf.predict(X_test).ravel()-y_test)).describe(),
                       np.abs(best_knn.predict(X_test).ravel() - y_test ).describe(),],axis = 1)


res_error.columns = ["Binomial Tree Absolute Error","CNN Absolute Error","Random Forest Absolute Error","KNN Absolute Error"]
res_error
```

    485/485 [==============================] - 0s 908us/step





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Binomial Tree Absolute Error</th>
      <th>CNN Absolute Error</th>
      <th>Random Forest Absolute Error</th>
      <th>KNN Absolute Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15508.000000</td>
      <td>15508.000000</td>
      <td>15508.000000</td>
      <td>15508.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.192882</td>
      <td>16.865260</td>
      <td>17.565473</td>
      <td>17.039533</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.810135</td>
      <td>27.633330</td>
      <td>27.825774</td>
      <td>28.158896</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000103</td>
      <td>0.000018</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.749303</td>
      <td>0.927289</td>
      <td>1.313031</td>
      <td>0.710667</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.469478</td>
      <td>4.053897</td>
      <td>5.481253</td>
      <td>4.335000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>17.896939</td>
      <td>22.537766</td>
      <td>22.226330</td>
      <td>21.908333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>369.079773</td>
      <td>350.456573</td>
      <td>338.299600</td>
      <td>341.098000</td>
    </tr>
  </tbody>
</table>
</div>



By considering the median for the absolute errors for each model, we can see that those of the machine learing models are smaller than that from the benchmark model. This implies that for at least half of the dataset, the absolute error from the machine learing models are smaller than that from the benchmark.
However, if we consider other quantitle, we can see that there are not much different. But for most part, the machine learning models have smaller errors.

### Stock Price versus Predicted Call Price

By the RMSE on the testing set, we choose the CNN to be the best model.
We visualize $C_0$ as a function of strike price $K$ and the stock price $S_0$.


```python
res_pt = pd.concat([X_test[['t0']].set_index(np.arange(0,len(X_test))).rename(columns = {"t0":"stock price"}),
                    X_test[['K']].set_index(np.arange(0,len(X_test))).rename(columns = {"K":"K"}),
            pd.DataFrame({"Call Price from the dataset": y_test.to_numpy().ravel()}),
           pd.DataFrame({"Call Price from Binomial Tree": bm_list}),
           pd.DataFrame({"Call Price from CNN": model_conv.predict(X_test).ravel()}),
          pd.DataFrame({"lower bound":
              [np.max([lb,0]) for lb in X_test['t0'] - X_test['K']*np.exp(-X_test['r'] *X_test['T'])]})], axis = 1)
```

    485/485 [==============================] - 1s 1ms/step



```python
# Percentage of call price greater than 120
np.sum(y_test.to_numpy() >= 120)/len(y_test)
```




    0.06970595821511479




```python
fig,axes = plt.subplots(1,3, figsize = (21,8),subplot_kw={'projection': '3d'})
axes[0].scatter(res_pt['stock price'], res_pt["K"],res_pt["Call Price from the dataset"], s= 1)
axes[0].set_xlabel("$S_0$")
axes[0].set_ylabel("$K$")
axes[0].set_zlabel("$C_0$")
axes[0].view_init(elev=20., azim=145, roll=0)
axes[0].set_title("True")
axes[0].set_zlim([0,320])

axes[1].scatter(res_pt['stock price'], res_pt["K"],res_pt["Call Price from Binomial Tree"], s= 1)
axes[1].set_xlabel("$S_0$")
axes[1].set_ylabel("$K$")
axes[1].set_zlabel("$C_0 $")
axes[1].view_init(elev=20., azim=145, roll=0)
axes[1].set_title("Binomial Tree")
axes[1].set_zlim([0,320])


axes[2].scatter(res_pt['stock price'], res_pt["K"],res_pt["Call Price from CNN"], s= 1)
axes[2].set_xlabel("$S_0$")
axes[2].set_ylabel("$K$")
axes[2].set_zlabel("$C_0 $")
axes[2].set_title("CNN")
axes[2].view_init(elev=20., azim=145, roll=0)
axes[2].set_zlim([0,320])


plt.show()
```



![png](/assets/output_178_0.png)



By considering only two factors, $S_0$ and $K$, we can see some paterrns in the corresponding option price. All the three plots look similar, indicating the accuracy of the binomial tree and the Convolutional Neural Network to price options. However, in the convolutional neural network, we can see that the call prices are dense around the small values similar to the actual price, however it did not well capture high call prices. For the benchmark, the prices are more sparse compared to both actual price and the CNN.

Now, we visualize an option price as a function of stock price, fixing other variable.

We fix other variable by choosing those variables from the dataset.

Then, we compare an option price as a function of $S_0$ from Binomial Tree and the best model.

Remark that, the machine learning models require additional parameters which are stock prices 8 days in the past. We cannot fix these stock prices; otherwise, the underlying price is unrealistic (eg. stock price at 0 is 500, while those at -1, -2, are around 200).

For each $S_0$, we find relevant $S_{-1}, S_{-2}$, ..., in the dataset.

We also compute a upperbound and lowerbound as shown below.


The upper and lower bound for American option (for non-dividend) can be computed by
$$\max(S_t - K e^{-r (T-t)},0)\leq C_t \leq S_t$$


```python
# Choose only S_0_x (rounded) that we use to train model
S_0_x = np.unique(np.round(spy_use['t0']) )
binomial_vs_s0 = [american_call_price(xs0, sample_fixed['K'][0], sigma = sample_fixed['hist_vol'][0], t = sample_fixed['T'][0], r = sample_fixed['r'][0], q = sample_fixed['q'][0], N = 30 ) for xs0 in S_0_x ]
spy_extended = spy_use.copy()
spy_extended['round_t0'] = np.round(spy_use['t0'])
s0_dataset = []
for xs0 in S_0_x:
    s0_dataset.append(np.concatenate([sample_fixed.drop(columns = ["call_last",'t0']).to_numpy()[0,:5], spy_extended[spy_extended['round_t0'] == xs0].sample(1).iloc[0,:9].ravel()]))
s0_dataset = np.array(s0_dataset)
cnn_vs_s0 = model_conv.predict(s0_dataset).ravel()
```

    7/7 [==============================] - 0s 1ms/step



```python
max_bound = S_0_x
min_bound = [np.max([xs0 - sample_fixed['K'][0] * np.exp(-sample_fixed['r'][0]*sample_fixed['T'][0]),0]) for xs0 in S_0_x ]
```


```python
plt.plot(S_0_x ,binomial_vs_s0,label = 'binomial')
plt.plot(S_0_x ,cnn_vs_s0,label = 'CNN')
plt.plot(S_0_x ,max_bound,label = 'max bound')
plt.plot(S_0_x ,min_bound,label = 'min bound',ls = 'dashed')
plt.ylim([0,100])
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f844f12e160>





![png](/assets/output_184_1.png)



The plot aligns with the previous plot, suggesting consistency. However, it appears that the CNN model does not accurately capture the high call prices. The generated call prices are observed to be outside the lower bound for high underlying price, creating the potential for arbitrage opportunities. Therefore, it is crucial to exercise caution and conduct a thorough examination of the model before implementing it. Additionally, fine-tuning or training the CNN model with a larger and more comprehensive dataset could potentially improve its performance and yield a better model.

## Conclusion


In this project, we have utilized machine learning models to price American options based on the SPY dataset. Our findings indicate that all of the machine learning models outperform the traditional binomial model for both the testing and training sets. Among these models, CNN, random forest, and KNN show promise, as their testing and training losses are relatively low.

However, it is worth noting that even though CNN performs the best overall, it struggles with accurately computing high call prices. Therefore, further tuning and dataset preparation might be necessary to enhance its performance in this regard.

In conclusion, machine learning models demonstrate the ability to capture the relationship between financial information and option pricing. Through the utilization of the RMSE metric, these models outperform the traditional benchmark model. Nonetheless, additional tunin

## Acknoledgement

Thanks all people who gathered the data and publicly publish them online. Thank you Dr. Kevin Lu for giving suggestions on American option pricing models.

## References

Culkin, Robert, and Sanjiv R. Das. Machine Learning in Finance: The Case of Deep Learning for Option Pricing, 2 Aug. 2017, srdas.github.io/Papers/BlackScholesNN.pdf.

French, Kenneth R. Kenneth R. French - Data Library, mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html. Accessed 3 June 2023.

Graupe, Kyle. “$spy Option Chains - Q1 2020 - Q4 2022.” Kaggle, 14 Mar. 2023,
www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022.

Lu, Kevin. “Machine Learning for Finance Lectures.” CFRM 421:Machine Learning for Finance . Seattle, The University of Washington.

Lu, Kevin. “Introduction to Financial Markets Lectures.” CFRM 415: Introduction to Financial Markets. Seattle, The University of Washington.

Mooney, Kevin, director. Implementing the Binomial Option Pricing Model in Python, YouTube, 15 Feb. 2021, https://www.youtube.com/watch?v=d7wa16RNRCI. Accessed 5 June 2023.

SPY Dividend Yield, ycharts.com/companies/SPY/dividend_yield. Accessed 3 June 2023.
