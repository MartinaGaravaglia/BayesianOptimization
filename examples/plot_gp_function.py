#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os 
# current working directory
path = os.getcwd()
# parent directory
parent = os.path.join(path, os.pardir)
sys.path.append(os.path.abspath(parent))
from bayes_opt1 import BayesianOptimization
from bayes_opt1 import UtilityFunction

def plot_gp(optimizer1, optimizer2, optimizer3, x, target):
    x = x.reshape(-1,1)
    y = target(x)
    
    def posterior(optimizer, x_obs, y_obs, grid):
        optimizer._gp.fit(x_obs, y_obs)
        mu, sigma = optimizer._gp.predict(grid, return_std=True)
        return mu, sigma
    
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer1.space)
    #fig.suptitle(
        #'Gaussian Process and Utility Function After {} Steps'.format(steps),
        #fontdict={'size':30}
    #)
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 2]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
   
    
    x_obs1 = np.array([[res["params"]["x"]] for res in optimizer1.res])
    y_obs1 = np.array([res["target"] for res in optimizer1.res])
    x_obs2 = np.array([[res["params"]["x"]] for res in optimizer2.res])
    y_obs2 = np.array([res["target"] for res in optimizer2.res])
    x_obs3 = np.array([[res["params"]["x"]] for res in optimizer3.res])
    y_obs3 = np.array([res["target"] for res in optimizer3.res])
    
    mu1, sigma1 = posterior(optimizer1, x_obs1, y_obs1, x)
    mu2, sigma2 = posterior(optimizer2, x_obs2, y_obs2, x)
    mu3, sigma3 = posterior(optimizer3, x_obs3, y_obs3, x)
    axis.plot(x, y, linewidth=3, label='Target')
    #axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu1, '--', color='purple', label='Prediction ucb')
    axis.plot(x, mu2, '--', color='green', label='Prediction ei')
    axis.plot(x, mu3, '--', color='orange', label='Prediction poi')
    
    #axis.axvline(x=x[np.argmax(y)], linestyle=':')
    #axis.fill(np.concatenate([x, x[::-1]]), 
              #np.concatenate([mu1 - 1.9600 * sigma1, (mu1 + 1.9600 * sigma1)[::-1]]),
        #alpha=.6, fc='c', ec='None',color='purple', label='95% confidence interval')
    
    fig.suptitle(
        'Utility Functions After {} Steps'.format(steps-2),
        fontdict={'size':50}
    )
    
    axis.set_xlim((min(x), max(x)))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function_ucb = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility_ucb = utility_function_ucb.utility(x, optimizer1._gp, max(np.array([res["target"] for res in optimizer1.res])))
    
    utility_function_ei = UtilityFunction(kind="ei", kappa=5, xi=0)
    utility_ei = utility_function_ei.utility(x, optimizer2._gp, max(np.array([res["target"] for res in optimizer2.res])))
    
    utility_function_poi = UtilityFunction(kind="poi", kappa=5, xi=0)
    utility_poi = utility_function_poi.utility(x, optimizer3._gp, max(np.array([res["target"] for res in optimizer3.res])))
    
    
    # UCB
    acq.plot(x, utility_ucb, label='UCB', color='purple')
    acq.plot(x[np.argmax(utility_ucb)], np.max(utility_ucb), '*', markersize=15, markerfacecolor='purple', markeredgecolor='k', markeredgewidth=1)
    
    # EI
    acq.plot(x, utility_ei, label='EI', color='green')
    acq.plot(x[np.argmax(utility_ei)], np.max(utility_ei), '*', markersize=15, markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
    
    # Poi
    acq.plot(x, utility_poi, label='POI', color='orange')
    acq.plot(x[np.argmax(utility_poi)], np.max(utility_poi), '*', markersize=15, markerfacecolor='orange', markeredgecolor='k', markeredgewidth=1)

    
    
    acq.set_xlim((min(x), max(x)))
    acq.set_ylim((min(min(utility_ucb), min(utility_ei), min(utility_poi)) - 0.5, max(max(utility_ucb), max(utility_ei), max(utility_poi)) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    

    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    plt.show()
    
    
    
def plot_convergence(optimizer1,optimizer2,optimizer3, x, target):
    x = x.reshape(-1,1)
    y = target(x)
    

    it=20
    tar1=np.zeros(it)
    tar2=np.zeros(it)
    tar3=np.zeros(it)
    point1=np.zeros(it)
    point2=np.zeros(it)
    point3=np.zeros(it)

        
    for i in range(it):
        utility_function1 = UtilityFunction(kind='ucb', kappa=5, xi=0)
        utility_function2 = UtilityFunction(kind='ei', kappa=5, xi=0)
        utility_function3 = UtilityFunction(kind='poi', kappa=5, xi=0)
        utility1 = utility_function1.utility(x, optimizer1._gp, max(np.array([res["target"] for res in optimizer1.res])))
        utility2 = utility_function2.utility(x, optimizer2._gp, max(np.array([res["target"] for res in optimizer2.res])))
        utility3 = utility_function3.utility(x, optimizer3._gp, max(np.array([res["target"] for res in optimizer3.res])))
        point1[i] = optimizer1.suggest(utility_function1)['x']
        point2[i] = optimizer2.suggest(utility_function2)['x']
        point3[i] = optimizer3.suggest(utility_function3)['x']
        tar1[i]= target(point1[i])
        tar2[i]= target(point2[i])
        tar3[i]= target(point3[i])
        optimizer1.register(params=point1[i], target=tar1[i])
        optimizer2.register(params=point2[i], target=tar2[i])
        optimizer3.register(params=point3[i], target=tar3[i])

    
    fig = plt.figure(figsize=(13, 6))
    
    acq = plt.subplot()
    
    fig.suptitle(
        'Convergences to the optimum after 20 iterations',
        fontdict={'size':50}
    )
    
    
    steps = len(optimizer1.space)
  
    
    num_iter=np.arange(1,it+1)
   
    acq.plot(num_iter, point1, '*',markersize=15,markerfacecolor='purple', markeredgecolor='k', markeredgewidth=1,label='UCB',linestyle='solid',color='purple')
    acq.plot(num_iter, point2, '*',markersize=15,markerfacecolor='green', markeredgecolor='k', markeredgewidth=1,label='EI',linestyle='solid',color='green')
    acq.plot(num_iter, point3, '*',markersize=15,markerfacecolor='orange', markeredgecolor='k', markeredgewidth=1,label='PoI',linestyle='solid',color='orange')
    acq.set_ylim((min(min(point1),min(point2),min(point3)) - 0.5, max(max(point1),max(point2),max(point3)) + 0.5))
    acq.axhline(y=x[np.argmax(target(x))], linestyle=':', label='Optimum to be achieved')
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    
    # Integer on the x axes
    a = range(0,21)
    acq.set_xticks(a)
    acq.set_yticks(range(-2,11))
    plt.xlabel('Number of iterations')
    plt.ylabel('Suggested x')
    
    
    plt.show()


    
    
    
