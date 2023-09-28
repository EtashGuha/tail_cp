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
import os
import pickle

def get_posterior(args, X_train, y_train):
    beta_post = None
    intercept_post = None
    b_post = None
    sigma_post = None
    folder_path = f"conformal_bayes_code/{args.dataset_name}_{args.seed}/post_samples"
    if os.path.exists(folder_path):
        beta_post = jnp.load(f"{folder_path}/beta_post.npy")
        intercept_post = jnp.load(f"{folder_path}/intercept_post.npy")
        b_post = jnp.load(f"{folder_path}/b_post.npy")
        sigma_post = jnp.load(f"{folder_path}/sigma_post.npy")
    else:
        beta_post, intercept_post, b_post, sigma_post = fit_mcmc_laplace(X_train, y_train, args)
        if not os.path.exists("conformal_bayes_code/{}_{}".format(args.dataset_name, args.seed)):
            os.mkdir("conformal_bayes_code/{}_{}".format(args.dataset_name, args.seed))
            os.mkdir("conformal_bayes_code/{}_{}/post_samples".format(args.dataset_name, args.seed))
        total_save_path = "conformal_bayes_code/{}_{}/post_samples".format(args.dataset_name, args.seed)
        np.save(f"{total_save_path}/beta_post", beta_post)
        np.save(f"{total_save_path}/intercept_post", intercept_post)
        np.save(f"{total_save_path}/b_post", b_post)
        np.save(f"{total_save_path}/sigma_post", sigma_post)
    return beta_post, intercept_post, b_post, sigma_post

## CONFORMAL FROM MCMC SAMPLES ##
### JAX IMPLEMENTATION
@jit #compute rank (unnormalized by n+1)
def compute_rank_IS(logp_samp_n,logwjk):
    n= jnp.shape(logp_samp_n)[0] #logp_samp_n is B x n
    n_plot = jnp.shape(logwjk)[0]
    rank_cp = jnp.zeros(n_plot)
    #compute importance sampling weights and normalizing
    wjk = jnp.exp(logwjk)
    Zjk = jnp.sum(wjk,axis = 1).reshape(-1,1)

    #compute predictives for y_i,x_i and y_new,x_n+1
    p_cp = jnp.dot(wjk/Zjk, jnp.exp(logp_samp_n).reshape(1, len(logp_samp_n)))
    p_new = jnp.sum(wjk**2,axis = 1).reshape(-1,1)/Zjk

    #compute nonconformity score and sort
    pred_tot = jnp.concatenate((p_cp,p_new),axis = 1)
    rank_cp = np.sum(pred_tot <= pred_tot[:,-1].reshape(-1,1),axis = 1)
    return rank_cp


#compute region of grid which is in confidence set
@jit
def compute_cb_region_IS(alpha,logp_samp_n,logwjk): #assumes they are connected
    n= jnp.shape(logp_samp_n)[0]#logp_samp_n is B x n
    rank_cp = compute_rank_IS(logp_samp_n,logwjk)
    region_true =rank_cp > alpha*(n+1)
    return region_true
## ##

## DIAGNOSE IMPORTANCE WEIGHTS ##
@jit #compute ESS/var
def diagnose_is_weights(logp_samp_n,logwjk):
    n= jnp.shape(logp_samp_n)[1] #logp_samp_n is B x n
    n_plot = jnp.shape(logwjk)[0]
    rank_cp = jnp.zeros(n_plot)

    #compute importance sampling weights and normalizing
    logwjk = logwjk.reshape(n_plot,-1, 1)
    logZjk = logsumexp(logwjk,axis = 1)

    #compute predictives for y_i,x_i and y_new,x_n+1
    logp_new = logsumexp(2*logwjk,axis = 1)-logZjk

    #compute ESS
    wjk = jnp.exp(logwjk - logZjk.reshape(-1,1,1))
    ESS = 1/jnp.sum(wjk**2,axis = 1)

    #compute variance for p_new
    var = np.sum(wjk**2*(wjk - jnp.exp(logp_new).reshape(-1,1,1))**2,axis = 1)
    return ESS, var
###
#Laplace prior PyMC3 model
def fit_mcmc_laplace(X_train,y_train, args):
    B=100
    with pm.Model() as model:
        p = np.shape(X_train)[1]
        #Laplace
        b = pm.Gamma('b',alpha = 1,beta = 1)
        beta = pm.Laplace('beta',mu = 0, b = b,shape = p)
        intercept = pm.Flat('intercept')
        if False:
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
    alpha = 0.1
    rep = np.shape(beta_post)[0]
    n_test = np.shape(X_val)[0]

    coverage_cb = np.zeros((rep,n_test))
    coverage_cb_exact = np.zeros((rep,n_test)) #avoiding grid effects

    length_cb = np.zeros((rep,n_test))

    region_cb = np.zeros((rep,n_test,np.shape(y_plot)[0]))
    if os.path.exists("saved_results/{}_{}/cb.pkl".format(args.dataset_name, args.seed)):
        with open("saved_results/{}_{}/cb.pkl".format(args.dataset_name, args.seed), "rb") as f:
            coverage_cb, length_cb = pickle.load(f)
    else:
        for j in tqdm(range(rep)):
            dy = y_plot[1] - y_plot[0]

            #Conformal Bayes
            @jit #normal loglik from posterior samples
            def normal_loglikelihood(y,x):
                return norm.logpdf(y,loc = jnp.dot(beta_post[j],x.transpose())+ intercept_post[j],scale = sigma_post[j]) #compute likelihood samples

            logp_samp_n = normal_loglikelihood(y_train, X_train)
            logwjk = normal_loglikelihood(y_plot.reshape(-1,1,1),X_val)
            logwjk_test = normal_loglikelihood(y_val, X_val).reshape(1,-1,n_test)

            for i in (range(n_test)):
                region_cb[j,i] = compute_cb_region_IS(alpha,logp_samp_n,logwjk[:,:,i])
                coverage_cb[j,i] = region_cb[j,i,np.argmin(np.abs(y_val[i]-y_plot))] #grid coverage
                length_cb[j,i] = np.sum(region_cb[j,i])*dy
            #compute exact coverage to avoid grid effects
            for i in (range(n_test)):
                coverage_cb_exact[j,i] = compute_cb_region_IS(alpha,logp_samp_n,logwjk_test[:,:,i]) #exact coverage

        print("---- RESULTS -----")
        if not os.path.exists("saved_results/{}_{}".format(args.dataset_name, args.seed)):
            os.mkdir("saved_results/{}_{}".format(args.dataset_name, args.seed))
        with open("saved_results/{}_{}/cb.pkl".format(args.dataset_name, args.seed), "wb") as f:
            pickle.dump((coverage_cb_exact, length_cb), f)
    mean_coverage = np.mean(coverage_cb_exact)
    std_coverage = np.std(coverage_cb_exact)
    se_coverage = np.std(coverage_cb_exact)/np.sqrt(len(coverage_cb_exact))
    mean_length = np.mean(length_cb)
    std_length = np.std(length_cb)
    se_length = np.std(length_cb)/np.sqrt(len(length_cb))
    return mean_coverage, std_coverage, mean_length, std_length, se_coverage, se_length
