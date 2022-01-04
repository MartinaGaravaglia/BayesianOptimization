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

def plot_gp(optimizer, x, target):
    x = x.reshape(-1,1)
    y = target(x)
    
    def posterior(optimizer, x_obs, y_obs, grid):
        optimizer._gp.fit(x_obs, y_obs)
        mu, sigma = optimizer._gp.predict(grid, return_std=True)
        return mu, sigma
    
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 2]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
   
    
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    #axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.axvline(x=x[np.argmax(y)], linestyle=':')
    #axis.fill(np.concatenate([x, x[::-1]]), 
              #np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        #alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((min(x), max(x)))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function_ucb = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility_ucb = utility_function_ucb.utility(x, None, optimizer._gp, 0)
    
    utility_function_ei = UtilityFunction(kind="ei", kappa=5, xi=0)
    utility_ei = utility_function_ei.utility(x, None, optimizer._gp, 0)
    
    utility_function_poi = UtilityFunction(kind="poi", kappa=5, xi=0)
    utility_poi = utility_function_poi.utility(x, None, optimizer._gp, 0)
    
    utility_function_kg = UtilityFunction(kind="kg", kappa=5, xi=0)
    utility_kg = utility_function_kg.utility(x, optimizer, optimizer._gp, 0)
    
    # UCB
    acq.plot(x, utility_ucb, label='UCB', color='purple')
    acq.plot(x[np.argmax(utility_ucb)], np.max(utility_ucb), '*', markersize=15, markerfacecolor='purple', markeredgecolor='k', markeredgewidth=1)
    
    # EI
    acq.plot(x, utility_ei, label='EI', color='green')
    acq.plot(x[np.argmax(utility_ei)], np.max(utility_ei), '*', markersize=15, markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
    
    # Poi
    acq.plot(x, utility_poi, label='POI', color='blue')
    acq.plot(x[np.argmax(utility_poi)], np.max(utility_poi), '*', markersize=15, markerfacecolor='blue', markeredgecolor='k', markeredgewidth=1)
    
    # KG
    acq.plot(x, utility_kg, label='KG', color='orange')
    acq.plot(x[np.argmax(utility_kg)], np.max(utility_kg), '*', markersize=15, markerfacecolor='orange', 
             markeredgecolor='k', markeredgewidth=1)
    
    
    acq.set_xlim((min(x), max(x)))
    acq.set_ylim((min(min(utility_kg), min(utility_ucb), min(utility_ei), min(utility_poi)) - 0.5, max(max(utility_kg),   max(utility_ucb), max(utility_ei), max(utility_poi)) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    #plt.plot(x, y, color='grey', label='Objective function', linewidth=4, linestyle=':' )
    #plt.axvline(x=x[np.argmax(y)], linestyle=':')
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    plt.show()
    
    
    
def plot_convergence(optimizer, x, target, kind):
    x = x.reshape(-1,1)
    y = target(x)
    
    #def suggest(utility, target, optimizer):
        #next_point = optimizer.suggest(utility)
        #target = target(**next_point)
        #optimizer.register(params=next_point, target=target)
    
        #return target, next_point
        #print(optimizer.max)
    it=20
    tar=np.zeros(it)
    point=np.zeros(it)
    opt=None
    if kind=='kg':
        opt=optimizer
        
    for i in range(it):
        utility_function = UtilityFunction(kind=kind, kappa=5, xi=0)
        utility = utility_function.utility(x, opt, optimizer._gp, 0)
        point[i] = optimizer.suggest(utility_function)['x']
        tar[i]= target(point[i])
        optimizer.register(params=point[i], target=tar[i])
      
    #print(point)
    #print(tar)
    #utility_function_kg = UtilityFunction(kind="kg", kappa=5, xi=0)
    #utility_kg = utility_function_kg.utility(x, optimizer, optimizer._gp, 0)
    #target_kg, point_kg = suggest(utility_function_kg, target, optimizer)
    
    fig = plt.figure(figsize=(25, 15))
    
    acq = plt.subplot()
    
    #kg_plt = plt.subplot(gs[])
    steps = len(optimizer.space)
    fig.suptitle(
        'Chosen points After {} Steps '.format(steps),
        fontdict={'size':30}
    )
    
    # UCB
    acq.plot(point, tar, '*',markersize=15,markerfacecolor='purple', markeredgecolor='k', markeredgewidth=1,label='UCB')
    acq.set_xlim((min(point)-0.5,max(point)+0.5))
    acq.set_ylim((min(tar) - 0.5, max(tar) + 0.5))
    acq.axvline(x=x[np.argmax(y)], linestyle=':')
    for i in range(it):
        acq.annotate(str(i+1), # this is the text
                     (point[i],tar[i]), # these are the coordinates to position the label
                     textcoords="offset points",
                     xytext=(0,10),# how to position the text
                     ha='center',
                     size=30) # horizontal alignment can be left, right or center
    #acq.plot(x[np.argmax(utility_ucb)], np.max(utility_ucb), '*', markersize=15, markerfacecolor='purple', markeredgecolor='k', markeredgewidth=1)
    """  
    # EI
    ei_plt.plot(point_ei, target_ei, '*',markersize=15,markerfacecolor='green', markeredgecolor='k', markeredgewidth=1,label='EI')
    ei_plt.set_xlim((-10,10))
    ei_plt.set_ylim((min(min(utility_ucb), min(utility_ei), min(utility_poi)) - 0.5, max(max(utility_ucb), max(utility_ei), max(utility_poi)) + 0.5))
    ei_plt.axvline(x=x[np.argmax(y)], linestyle=':')
     
    #acq.plot(x[np.argmax(utility_ei)], np.max(utility_ei), '*', markersize=15, markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
    
    # Poi
    poi_plt.plot(point_poi, target_poi, '*',markersize=15,markerfacecolor='blue', markeredgecolor='k', markeredgewidth=1,label='POI')
    poi_plt.set_xlim((-10,10))
    poi_plt.set_ylim((min(min(utility_ucb), min(utility_ei), min(utility_poi)) - 0.5, max(max(utility_ucb), max(utility_ei), max(utility_poi)) + 0.5))
    poi_plt.axvline(x=x[np.argmax(y)], linestyle=':')
    """
    plt.show()

    #acq.plot(x[np.argmax(utility_poi)], np.max(utility_poi), '*', markersize=15, markerfacecolor='blue', markeredgecolor='k', markeredgewidth=1)
 
    
  
    # KG
    #kg_plt.plot(point_kg, target_kg, label='KG', color='orange')
    #acq.plot(x[np.argmax(utility_kg)], np.max(utility_kg), '*', markersize=15, markerfacecolor='orange', 
            # markeredgecolor='k', markeredgewidth=1)
    
    
    
    
