# 50日目 Time Series Data processing 5 Day48 Time Series Data processing 5

本日の目標は
1. 定常過程の時系列モデルの比較
2. 定常過程の時系列モデルMA, ARMAを実装してみる


## Step 1: 定常過程の時系列モデルの比較

### AR(q)  
ACF(autocorrelation function):だんだん0に近く　→ PACFでlag期間を探す  
PACF(partial autocorrelation function):p期で終わる  

### MA(p)  
ACF(autocorrelation function):だんだん0に近く　→ q期で終わる  
PACF(partial autocorrelation function):だんだん0に近く  

### ARMA(p, q)  
ACF(autocorrelation function):だんだん0に近く　→ q期で終わる  
PACF(partial autocorrelation function):だんだん0に近く  


```python
# 定常過程を確定になったら、定常過程用のモデルが使えるようになる
# ARモデルを実装

# 絵を描くfunction
def tsplot(y, lags=None, figsize=(10, 8), style='bmh',title=''):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


# MA(1)のモデルを実装
n = int(1000)
# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.8])
# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
limit=12
_ = tsplot(ma1, lags=limit,title="MA(1) process")
# MA(1) : ACF lag=1 で終わり

# MA(2)のモデルを実装
n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6, 0.4])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma3, lags=12,title="MA(2) process")
# MA(2) : ACF lag=2 で終わり

# ARMA(2,2)のモデルを実装
max_lag = 12

n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit

alphas = np.array([0.8, -0.65])
betas = np.array([0.5, -0.7])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
_ = tsplot(arma22, lags=max_lag,title="ARMA(2,2) process")
```


### 参考資料
AR/MAモデル  
http://muddydixon.hatenablog.com/entry/2013/03/23/050131
