import time
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm
from sklearn.linear_model import LassoCV,Lasso
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import time
import scipy as sp
import pymc3 as pm
from helper_methods import compute_cb_region_IS

#Laplace prior PyMC3 model
def fit_mcmc_laplace(X_train,y_train, args):
    B=2000
    with pm.Model() as model:
        p = np.shape(X_train)[1]
        #Laplace
        b = pm.Gamma('b',alpha = 1,beta = 1)
        beta = pm.Laplace('beta',mu = 0, b = b,shape = p)
        intercept = pm.Flat('intercept')
        if misspec == True:
            sigma = pm.HalfNormal("sigma", sigma = 0.02) ## misspec prior
        else:
            sigma = pm.HalfNormal("sigma", sigma = 1) 
        obs = pm.Normal('obs',mu = pm.math.dot(X_train,beta)+ intercept,sigma = sigma,observed=y_train)

        trace = pm.sample(B,random_seed = args.seed, chains = 4)
    beta_post = trace['beta']
    intercept_post = trace['intercept'].reshape(-1,1)
    sigma_post = trace['sigma'].reshape(-1,1)
    b_post = trace['b'].reshape(-1,1)
    print(np.mean(sigma_post)) #check misspec.

    return beta_post,intercept_post,b_post,sigma_post

#Main run function for sparse regression
def run_cb(X_train, y_train, X_val, y_val, beta_post, intercept_post, sigma_post, args):
    # there's gotta be some sort of model path thing for the posterior distributions if we want to save them
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    y_plot = np.linspace(np.min(y_train) - 2, np.max(y_train) + 2,100) # not entirely sure what this is for

    #Initialize
    alpha = 0.2
    rep = np.shape(beta_post)[0]
    n_test = np.shape(X_val)[0]

    coverage_cb = np.zeros((rep,n_test))
    coverage_cb_exact = np.zeros((rep,n_test)) #avoiding grid effects

    length_cb = np.zeros((rep,n_test))

    region_cb = np.zeros((rep,n_test,np.shape(y_plot)[0]))


    for j in tqdm(range(rep)):
        dy = y_plot[1] - y_plot[0]

        #Conformal Bayes
        @jit #normal loglik from posterior samples
        def normal_loglikelihood(y,x):
            return norm.logpdf(y,loc = jnp.dot(beta_post[j],x.transpose())+ intercept_post[j],scale = sigma_post[j]) #compute likelihood samples

        logp_samp_n = normal_loglikelihood(y_train, X_train)
        logwjk = normal_loglikelihood(y_plot.reshape(-1,1,1),X_val)
        logwjk_test = normal_loglikelihood(y_train, X_val).reshape(1,-1,n_test)

        for i in (range(n_test)):
            region_cb[j,i] = compute_cb_region_IS(alpha,logp_samp_n,logwjk[:,:,i])
            coverage_cb[j,i] = region_cb[j,i,np.argmin(np.abs(y_val[i]-y_plot))] #grid coverage
            length_cb[j,i] = np.sum(region_cb[j,i])*dy
        #compute exact coverage to avoid grid effects
        for i in (range(n_test)):
            coverage_cb_exact[j,i] = compute_cb_region_IS(alpha,logp_samp_n,logwjk_test[:,:,i]) #exact coverage

    print("---- RESULTS -----")
    mean_coverage = np.mean(coverage_cb)
    std_coverage = np.std(coverage_cb)
    mean_length = np.mean(length_cb)
    std_length = np.std(length_cb)
    return mean_coverage, std_coverage, mean_length, std_length
