#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.ensemble import RandomForestRegressor
import scipy
import scipy.sparse as sparse
from scipy.stats import norm


def StockVol(histoPrice):
    """
    Compute the stock volatility under GBM using 1-year historical prices
    
    Inputs:
        histoPrice: an array of daily historical prices for one year
        
    Returns:
        histoVol: annualized historical volatility
    """
    
    logret = np.diff(np.log(histoPrice))
    sigma = np.sqrt(np.var(logret))
    histoVol = sigma*np.sqrt(252)  # annualize volatility
    
    return histoVol


def StockPath(n,sigma=0.2,S0=100,T=1,nump=252,r=0.01,delta = 0):
    """
    Generate n stock paths
    
    Inputs:
        n: number of paths generated
        sigma: volatility of the stock
        S0: current stock price
        T: terminal time in yearly unit
        nump: number of time periods
        r: interest rate
        delta: continuous dividend yield of the stock
        
    Returns:
        S: an array of stock paths
    """
    
    X = np.zeros((n,1+nump))
    X[:,0] = S0
    for i in range(len(X)):
        Z = np.random.normal(0, 1, nump)
        X[i,1:]=np.exp(sigma*np.sqrt(T/nump)*Z+(r-delta-sigma**2/2)*(T/nump))
    
    S = []   
    for i in range (n):
        S.append(np.cumprod(X[i,:]))
    
    return np.array(S)

def EurOptPrice(paths,K,r=0.01,T=1):
    """
    generate the European put option price through Monte Carlo method
    
    Inputs:
        paths: an array of stock paths
        K: strike price
        r: interest rate
        T: terminal time
        
    Returns:
        Payoff: discounted payoffs
        price: estimated price of the European put option
        variance: variance of discounted payoffs
        
    """
    
    Payoff = np.maximum(K-paths[:,-1],0)*np.exp(-r*T)
    price = np.mean(Payoff)
    variance = np.var(Payoff)
    
    return (Payoff,price,variance)

def AmeOptPrice(paths,K,r=0.01,T=1,nump = 252,delta = 0,sigma=0.2):
    """
    generate the American put option price without control variable
    
    Inputs:
        paths: an array of stock paths
        K: strike price
        r: interest rate
        T: terminal time
        nump: number of periods
        
    Returns:
        Payoff: discounted payoffs
        price: estimated price of the American put option
        variance: variance of discounted payoffs
        
    """
    deltaT = T/nump
    P = np.maximum(K-paths,0)  # payoffs if early exercise
    H = np.zeros(paths.shape)  # holding value
    V = np.zeros(paths.shape)  # value of the option
    
    H[:,-1] = P[:,-1]
    V[:,-1] = P[:,-1]
    
    # compute the expected payoff at termial time given S_(T-1) using one step monte carlo
    tmp = paths[:,-2]
    for i in range(len(paths)):
        tmp_Price = StockPath(100,sigma,tmp[i],deltaT,1,r,delta)
        tmp_payoff = np.maximum(K-tmp_Price[:,-1],0)*np.exp(-r*deltaT)
        H[i,-2] = np.mean(tmp_payoff)
    V[:,-2] = np.maximum(P[:,-2], H[:,-2])  # value of the option at t = T-1
    
    rf = RandomForestRegressor(n_estimators=30, n_jobs=-1)  #Define Random Forest Regressor 
    
    for i in range(2,len(V[0])):
        X = paths[:,-i].reshape(-1,1)
        Y = V[:,-i].reshape(-1,1)
        
        reg = rf.fit(X, Y.ravel())  # Polynomial regression (degree = 5)
        
        tmp = paths[:,-i-1]
        for j in range(len(paths)):
            tmp_Price = StockPath(100,sigma,tmp[j],deltaT,1,r,delta)
            tmp_V = rf.predict(tmp_Price.reshape(-1,1))*np.exp(-r*deltaT)
            H[j,-i-1] = np.mean(tmp_V)
        V[:,-i-1] = np.maximum(P[:,-i-1], H[:,-i-1])
 
    # Determine the optimal stopping time and payoffs
    Payoff = [0]*len(P)
    for i in range(len(P)):
        idx = np.where(P[i,:]> H[i,:])[0]
        if(len(idx) == 0):
            Payoff[i] = V[i,-1]*np.exp(-r*T)
        else:
            Payoff[i] = V[i,idx[0]]*np.exp(-r*idx[0]*deltaT)

    price = np.mean(Payoff)
    variance = np.var(Payoff)
    
    return(Payoff, price, variance)

def ContVariate(y,x,mu_x):
    """
    Implement the control variates method
    
    Inputs:
        y: an array of samples with unknown mean
        x: an array of samples with known or estimated mean
        mu_x: mean of random variable X
        
    Returns:
        y_hat: estimated mean of y
        y_hatVar: variance of the estimator
        
    """
    
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    y_var = np.var(y)
    corr = np.corrcoef(x,y)[0,1]
    
    beta = beta = np.cov(x,y)[0,1]/np.var(x)
    y_hat = y_bar+beta*(mu_x-x_bar)
    y_hatVar = (np.var(y)/len(y))*(1-corr**2)
    
    return(y_hat[0],y_hatVar)

def BSput(S0, K, T, r, sigma,q):
    """
    Compute true price of the European put using BS formula
    
    Inputs:
        S0: current stock price
        K: strike price
        T: termial time
        r: interest rate
        sigma: volatility
        q: continuous dividend
        
    Returns:
        price: price of the option
    """
    
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    price = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S0 * norm.cdf(-d1, 0.0, 1.0))
    
    return price

