

"""

Created on Tue April 30 08:24:00 2019

FINN 6212 - Spring 2018

Topic: Pricing Exotic options using Monte Carlo simulations

                                          

Author: Akshay D Patil

Instructor: Dr.Steven P Clark

"""



import math

import numpy as np

import scipy.stats as stat

#import scipy.optimize as sop

import os

os.chdir('/Users/patil/Desktop/Sem-IV/FINN 6212')



def BSM_put_Value(S0, K, T, r, sigma):

    ''' Calculates Black-Scholes-Merton European put option value.



    Parameters

    ==========

    S0 : float

        stock/index level at initial time

    K : float

        strike price

    T : float

        Number of years till maturity

    r : float

        constant, risk-less short rate

    sigma : float

        volatility



    Returns

    =======

    put_value : float

        European put present value at initial time

    '''

    

    d1 = -((np.log(S0 / K) + (r + 0.5 * sigma ** 2)

          * T) / (sigma * math.sqrt(T)))

    d2 = -(d1 - sigma * math.sqrt(T))

    

    put_value = np.exp(-r * T )* K * stat.norm.cdf(d2,0,1) -S0 * stat.norm.cdf(d1,0,1)  

    

    return put_value





def down_Out_put(S0,B,r,sigma,K,T):         

    ''' Calculates Black-Scholes-Merton Up and out barrier call option value.



    Parameters

    ==========

    S0 : float

        stock/index level at initial time

    K : float

        strike price

    B : float

        barrier price       

    T : float

        Number of years till maturity

    r : float

        constant, risk-less short rate

    sigma : float

        volatility



    Returns

    =======

    put_value : float

    down and out barrier put present value at initial time

    '''

    

    if K < B:

        raise ValueError ("Barrier value cannot be greater than strike price.")



    

    a = (B/S0) ** ((- 1+ (2 * r)) / sigma ** 2)

    b = (B/S0) ** (1 + (2 * r / sigma ** 2))

    

    d1 = ( np.log(S0/K) + (r + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))

    d2 = ( np.log(S0/K) + (r - 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))

    d3 = ( np.log(S0/B) + (r + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))    

    d4 = ( np.log(S0/B) + (r - 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))    

    d5 = ( np.log(S0/B) - (r - 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))    

    d6 = ( np.log(S0/B) - (r + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))    

    d7 = ( np.log((S0 * K)/B ** 2) - (r - 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))    

    d8 = ( np.log((S0 * K)/B ** 2) - (r + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))       

    

    p =  (K * np.exp(-r * T) * (stat.norm.cdf(d4,0,1) - stat.norm.cdf(d2,0,1)) - a * (stat.norm.cdf(d7,0,1) - stat.norm.cdf(d5,0,1))) -S0 * ((stat.norm.cdf(d3,0,1) - stat.norm.cdf(d1,0,1)) -b*(stat.norm.cdf(d8,0,1) - stat.norm.cdf(d6,0,1)))

    

    return p





def Stock_Paths(S0,mu,sigma,T,NSteps,NRepl):  

    ''' Generates the stock paths for different replications

    Parameters

    ==========

    S0 : float

        stock/index level at initial time

    mu : float

        constant, mean

    T : float

        Number of years till maturity

    NSteps : float

        constant, number of discrete steps  of T

    NRepl : float

        constant, number of replications of stock paths

    sigma : float

        volatility



    Returns

    =======

    spaths :vector

        Replications of stock paths for discrete steps of T 

    '''

    

    spaths = np.zeros((NRepl,1+NSteps))

    spaths[:,0] = S0

    dt = T/NSteps

    nudt = (mu- 0.5 * sigma ** 2) * dt

    sidt= sigma * np.sqrt(dt)

    for i in range(0,NRepl):

        for j  in range(1,NSteps+1):

            spaths[i,j] = spaths[i,j-1] * (np.exp(nudt + (sidt * np.random.normal(0,1))))

    return spaths





def Stock_Paths_AV(S0,mu,sigma,T,NSteps,NRepl):

    ''' Generates the two different stock paths for different replications

    Returns

    =======

    spaths_a, spaths_b :vector

        Replications of two differnt stock paths for discrete steps of T 

    '''

    spaths_a = np.zeros((NRepl,1+NSteps))

    spaths_a[:,0] = S0

    spaths_b = np.zeros((NRepl,1+NSteps))

    spaths_b[:,0] = S0

    dt = T/NSteps

    nudt = (mu- 0.5 * sigma ** 2) * dt

    sidt= sigma * np.sqrt(dt)

    for i in range(0,NRepl):

        for j  in range(1,NSteps+1):

            spaths_a[i,j] = spaths_a[i,j-1] * (np.exp(nudt + (sidt * np.random.normal(0,1))))         

            spaths_b[i,j] = spaths_b[i,j-1] * (np.exp(nudt + (sidt * np.random.normal(0,1))))

    return spaths_a, spaths_b 

   



     

def down_Out_put_Monte_Carlo(S0,K,r,T,sigma,B,NSteps,NRepl):

    ''' Calculates the down and out barrier put value using monte carlo simulations

    Returns

    =======

    mean, std error :float

       Option price and standard error 

    '''

    if K < B:

        raise ValueError ("Barrier value cannot be greater than strike price.")

        

    payoff = np.zeros((NRepl,1))

    ncrossed = 0

    for i in range(1,NRepl):

        mu = r

        path = Stock_Paths(S0,mu,sigma,T,NSteps,1)

        crossed = path >= B

        

        if crossed.sum() == 0 :

            payoff[i,0] = max(0, path[:,NSteps] - K)

        else:

            payoff[i,0] = 0

            ncrossed = ncrossed + 1

    mean, std = stat.norm.fit(np.exp(-r * T) * payoff)

    return mean, std







def down_Out_put_Antithetic(S0,B,K,r,T,sigma,NSteps,NRepl):

    ''' Calculates the down and out barrier put value using 

    monte carlo simulation and antithetic variates variance reduction technique

    Returns

    =======

    mean, std error :float

       Option price and standard error 

    '''

    if K < B:

        raise ValueError ("Barrier value cannot be greater than strike price.")

        

    payoff_a = np.zeros((NRepl,1))

    payoff_b = np.zeros((NRepl,1))

    

    for i in range(1,NRepl):

        mu=r

        path_a, path_b = Stock_Paths_AV(S0,mu,sigma,T,NSteps,1)

        

        crossed_a = path_a >= B

        

        if crossed_a.sum() == 0 :

            payoff_a[i,0] = max(0, path_a[:,NSteps] - K)

        crossed_b = path_b >= B

        

        if crossed_b.sum() == 0 :

            payoff_b[i,0] = max(0, path_b[:,NSteps] - K)

            

    payoff = (payoff_a + payoff_b) / 2

    mean, std = stat.norm.fit(np.exp(-r * T) * payoff)  

    return mean, std

    



    

def down_Out_Call_Control_Variate(S0,B,K,r,T,sigma,NSteps,NRepl,NPilot):

    ''' Calculates the down and out barrier put value using 

    monte carlo simulation and control variates variance reduction technique

    Returns

    =======

    mean, std error :float

       Option price and standard error 

    '''   

    if K < B:

        raise ValueError ("Barrier value cannot be greater than strike price.")

        

        put_value = BSM_put_Value(S0, K, T, r, sigma)

    

    #Vanilla Payoff variables

    

    payoff = np.zeros((NPilot,1))

    vanilla_payoff = np.zeros((NPilot,1))

    

    for i in range(0,NPilot):

        path = Stock_Paths(S0,r,sigma,T,NSteps,1)

        vanilla_payoff[i,:] = max(0, path[:,NSteps] - K)

        crossed = path >= B

        

        if crossed.sum() == 0 :

            payoff[i,:] = max(0, path[:,NSteps] - K)

            

    vanilla_payoff = np.exp(-r * T) * vanilla_payoff

    payoff = np.exp(-r * T) * payoff

    

    covar_vanilla = np.cov(vanilla_payoff,payoff,bias=True)

    var_vanilla = np.var(vanilla_payoff)

    corr = - covar_vanilla[0,1] / var_vanilla

    

    #New Payoff variables

    

    new_payoff = np.zeros((NRepl,1))

    new_vanilla_payoff = np.zeros((NRepl,1))

    for i in range(0,NRepl):

        path = Stock_Paths(S0,r,sigma,T,NSteps,1)

        new_vanilla_payoff[i,:] = max(0, path[:,NSteps] - K)

        crossed = path >= B

        

        if crossed.sum() == 0 :

            new_payoff[i] = max(0, path[:,NSteps] - K)

    

    new_vanilla_payoff = np.exp(-r * T) * new_vanilla_payoff

    new_payoff  = np.exp(-r * T) * new_payoff

    cv_payoff  = new_payoff + (corr * (new_vanilla_payoff - put_value))

    mean, std = stat.norm.fit(np.exp(-r * T) * cv_payoff)

    

    return mean, std

    





   
        

        

        