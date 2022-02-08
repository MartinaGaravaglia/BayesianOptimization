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


def normalization(utility):
    return (utility- min(utility)) / (max(utility) - min(utility))


def plot_gp(optimizers, x, target, params, n_init=2): 
    """
    params è un dizionario che per tutte le acqusition deve contenere le chiavi 'kappa' e 'xi'
    """
    x = x.reshape(-1,1)
    y = target(x)
    
    def posterior(optimizer, x_obs, y_obs, grid):
        optimizer._gp.fit(x_obs, y_obs)
        mu, sigma = optimizer._gp.predict(grid, return_std=True)
        return mu, sigma
    
    fig = plt.figure(figsize=(16, 10))
    steps = len(list(optimizers.values())[0].space)
  
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 2]) 
    axis = plt.subplot(gs[0])
    af = plt.subplot(gs[1])
    n = len(optimizers)
    colmap = plt.cm.get_cmap(name='rainbow', lut=n)
    
    x_obs = {}  # dizionario vuoto
    y_obs={}
    mu={}
    sigma={}
    utility_function={}
    utility={}
    for acq, optimizer in optimizers.items():  # acq è la stringa-chiave, optimizer è l'oggetto
        x_obs[acq] = np.array([[res["params"]["x"]] for res in optimizer.res])
        y_obs[acq]=  np.array([res["target"] for res in optimizer.res])
        mu[acq], sigma[acq] = posterior(optimizer, x_obs[acq], y_obs[acq], x)
        utility_function[acq] = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])
        utility[acq] = utility_function[acq].utility(x, optimizer._gp, optimizer._space.target.max())

    axis.plot(x, y, linewidth=3, label='Target')
    i=0 #indice per i colori
    for acq, mean in mu.items():
        axis.plot(x, mean, '--', color= colmap(i), label='Prediction {}'.format(acq) )
        i=i+1
    
    fig.suptitle(
        'Utility Functions After {} Steps'.format(steps-n_init),
        fontdict={'size':50}
    )
    
    axis.set_xlim((min(x), max(x)))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
 
    i=0
    for acq, util in utility.items():
        af.plot(x, normalization(util), label=acq, color= colmap(i))
        af.plot(x[np.argmax(util)], np.max(normalization(util)), '*', markersize=15, markerfacecolor=colmap(i), markeredgecolor='k', markeredgewidth=1)
        i=i+1

    af.set_xlim((min(x), max(x)))
    af.set_ylim((-0.5,1.5))
    af.set_ylabel('Utility', fontdict={'size':20})
    af.set_xlabel('x', fontdict={'size':20})
    

    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    af.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    plt.show()
    
    
    
def plot_convergence(optimizers, x, target, params, it=20):
    x = x.reshape(-1,1)
    y = target(x)
    
    tar={}
    points={}
    """
    tar1=np.zeros(it)
    tar2=np.zeros(it)
    tar3=np.zeros(it)
    point1=np.zeros(it)
    point2=np.zeros(it)
    point3=np.zeros(it)
    """
    utility_function={}
    for acq, optimizer in optimizers.items():  # acq è la stringa-chiave, optimizer è l'oggetto
        utility_function[acq] = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])
        
    """
    utility_function1 = UtilityFunction(kind='ucb', kappa = params['ucb']['kappa'], xi = params['ucb']['xi'])
    utility_function2 = UtilityFunction(kind='ei', kappa = params['ucb']['kappa'], xi = params['ei']['xi'])
    utility_function3 = UtilityFunction(kind='poi', kappa = params['ucb']['kappa'], xi = params['poi']['xi'])    
    """
    for acq, optimizer in optimizers.items():
            points[acq]=np.zeros(it)
            tar[acq]= np.zeros(it)
            
    for i in range(it):
        for acq, optimizer in optimizers.items():
            points[acq][i]=optimizer.suggest(utility_function[acq])['x']
            tar[acq][i]= target(points[acq][i])
            optimizer.register(params=points[acq][i], target=tar[acq][i])
        """
        point1[i] = optimizer1.suggest(utility_function1)['x']
        point2[i] = optimizer2.suggest(utility_function2)['x']
        point3[i] = optimizer3.suggest(utility_function3)['x']
        tar1[i]= target(point1[i])
        tar2[i]= target(point2[i])
        tar3[i]= target(point3[i])
        optimizer1.register(params=point1[i], target=tar1[i])
        optimizer2.register(params=point2[i], target=tar2[i])
        optimizer3.register(params=point3[i], target=tar3[i])
        """
    
    fig = plt.figure(figsize=(13, 6))
    
    af = plt.subplot()
    
    fig.suptitle(
        'Convergences to the optimum after {} iterations'.format(it),
        fontdict={'size':50}
    )
    
    
    steps = len(list(optimizers.values())[0].space)
    n = len(optimizers)
    colmap = plt.cm.get_cmap(name='rainbow', lut=n)
    
    num_iter=np.arange(1,it+1)
    
    i=0
    for acq in optimizers.keys():
        af.plot(num_iter, points[acq], '*',markersize=15,markerfacecolor=colmap(i), markeredgecolor='k', markeredgewidth=1,label=acq,linestyle='solid',color=colmap(i))
        i=i+1
    
 
    af.axhline(y=x[np.argmax(target(x))], linestyle=':', label='Optimum to be achieved')
    af.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
   
    
    # Integer on the x axes
    a = range(0,21)
    af.set_xticks(a)
    af.set_yticks(range(-2,11))
    plt.xlabel('Number of iterations')
    plt.ylabel('Suggested x')
    
    
    plt.show()


# Plot regret

def simple_regret(f_max, optimizer):
    """
    The simple regret rT = max{x∈X} f(x) − max{t∈[1,T]} f(x_t) measures the value of
the best queried point so far. 
    """
    return f_max-optimizer._space.target.max()

def plot_simple_regret(optimizers, x, target, params, dim, it=20):
    
    if dim > 1:
        inputs = []
        for col in range(dim):
            inputs.append(x[:,col])
        y = target(*inputs)


        f_max=max(y)

        tar={}
        points={}
        utility_function={}
        regrets={}


        for acq, optimizer in optimizers.items():  # acq è la stringa-chiave, optimizer è l'oggetto
            regrets[acq]=np.zeros(it)
            utility_function[acq] = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])

        for i in range(it):
            for acq, optimizer in optimizers.items():
                points[acq]=optimizer.suggest(utility_function[acq])
                tar[acq]= target(**points[acq])
                optimizer.register(params=points[acq], target=tar[acq])
                regrets[acq][i]= simple_regret(f_max, optimizer)
    else:
        x = x.reshape(-1,1)
        y = target(x)
        
        f_max=max(y)

        tar={}
        points={}
        utility_function={}
        regrets={}

        for acq, optimizer in optimizers.items():  # acq è la stringa-chiave, optimizer è l'oggetto
            points[acq]=np.zeros(it)
            tar[acq]= np.zeros(it)
            regrets[acq]=np.zeros(it)
            utility_function[acq] = UtilityFunction(kind=acq, kappa = params[acq]['kappa'], xi = params[acq]['xi'])

        for i in range(it):
            for acq, optimizer in optimizers.items():
                points[acq][i]=optimizer.suggest(utility_function[acq])['x']
                tar[acq][i]= target(points[acq][i])
                optimizer.register(params=points[acq][i], target=tar[acq][i])
                regrets[acq][i]= simple_regret(f_max, optimizer)

    
    fig = plt.figure(figsize=(13, 6))
    
    af = plt.subplot()
    
    fig.suptitle(
        'Simple regret after {} iterations'.format(it),
        fontdict={'size':50}
    )
    
    steps = len(list(optimizers.values())[0].space)
    n = len(optimizers)
    colmap = plt.cm.get_cmap(name='rainbow', lut=n)
    
    num_iter=np.arange(1,it+1)
    
    i=0
    for acq in optimizers.keys():
        af.plot(num_iter, regrets[acq], '*',markersize=15,markerfacecolor=colmap(i), markeredgecolor='k', markeredgewidth=1,label=acq,linestyle='solid',color=colmap(i))
        i=i+1

    af.set_ylim((-0.2,1))
    af.axhline(y=0, linestyle=':', label='0.0')
    af.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    
    # Integer on the x axes
    a = range(0,21)
    af.set_xticks(a)
    
    plt.xlabel('Number of iterations')
    plt.ylabel('Simple regret')
    
    
    plt.show()
    
    
    
