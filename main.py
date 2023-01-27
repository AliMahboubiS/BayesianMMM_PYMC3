import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import arviz as az
import pymc3 as pm

def geometric_adstock(x, theta, alpha,L):
    w = tt.as_tensor_variable([tt.power(alpha,tt.power(i-theta,2)) for i in range(L)])
    xx = tt.stack([tt.concatenate([tt.zeros(i), x[:x.shape[0] -i]]) for i in range(L)])
    return tt.dot(w/tt.sum(w), xx)

def coef_mul(x,b):
    return b * x

if __name__ == '__main__':
    
    X = pd.read_csv('MMM_test_data.csv',parse_dates=['start_of_week'])
    y = X['revenue'].values

    media = ["spend_channel_1", "spend_channel_2", "spend_channel_3", "spend_channel_4", "spend_channel_5", "spend_channel_6", "spend_channel_7"]

    print(X.info())

    target = "revenue"
    for feature in media:
        scaler = MinMaxScaler()
        original = X[feature].values.reshape(-1, 1)
        transformed = scaler.fit_transform(original)
        X[feature] = transformed

    dependent_transformation = None
    original = X[target].values
    X[target] = original / 100_000

    with pm.Model() as m:
        #var,      dist, pm.name,          params,  shape   
        alpha = pm.Beta('alpha', 3 , 3, shape = 7) 
        theta = pm.Uniform('theta', 0 , 12, shape = 7) 
        beta  = pm.HalfNormal('beta', 1, shape=7)
        tau    = pm.HalfNormal('intercept', 5) 
        noise = pm.InverseGamma('noise',   0.05,  0.005)
        

        computations = []
        idx = 0
        for col in media:   
            comp = coef_mul(x=geometric_adstock(x=X[col].values, 
                                                    alpha = alpha[idx],
                                                    theta= theta[idx],
                                                    L=1),
                            b=beta[idx])
            
            computations.append(comp)
            idx += 1

        y_hat = pm.Normal
        y_hat = pm.Normal('y_hat', mu= tau + sum(computations),
                    sigma=noise, 
                    observed=y)
        trace = pm.sample(init='adapt_diag',
                  return_inferencedata=True,
                  tune=10)
        
        trace_summary = az.summary(trace)
        
        az.plot_trace(trace, compact=True)
        plt.show()
