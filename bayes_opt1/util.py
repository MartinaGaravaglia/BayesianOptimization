import sys
import warnings
import numpy as np
import warnings
from pyDOE import lhs
import numdifftools as nd

from scipy.stats import norm
from scipy.optimize import minimize
import random
from pyDOE import lhs


def posterior(gp, x_obs, y_obs, grid):
        '''
        fits gaussian process on points (x_obs[i], y_obs[i]) and evaluates its mean on each point
        in grid. Notice that "fit" method implemented in sklearn library is very expensive.
        '''
        gp.fit(x_obs, y_obs)
        mu = gp.predict(grid, return_std=False)
        return mu
    
def find_mean_max(x_obs,y_obs,gp,grid,mean=None,std=None,i=None):
    '''
    If i is None, then it means we are looking for what the paper called \mu^{*}_{n}, otherwise we want to
    compute \mu^{*}_{n+1}.
    Remark: i=None <=> mean=None <=> std=None
    '''
    # isinstance(i, type(None))
    if i is None: # checks if i is None
        x_newobs = x_obs
        y_newobs = y_obs
    else:
        y=np.random.normal(loc=mean, scale=std)
        x_newobs=np.concatenate((x_obs, i.reshape(1,-1)), axis = 0)
        y_newobs=np.append(y_obs,y) 
    mu=posterior(gp, x_newobs, y_newobs, grid)
    return (max(mu),np.argmax(mu))
    
def grid_construction(pbounds, n_grid):
    
    dim = pbounds.shape[0]
    init = [np.zeros(n_grid)] * dim
    
    for i in range(dim): 
        init[i] = np.linspace(pbounds[i,0], pbounds[i,1], n_grid) # works \forall p

    grid = np.meshgrid(*init) # list with p matrices n_grid x n_grid
    for g in range(len(grid)):
        grid[g] = grid[g].reshape(-1,1)
    grid = np.stack(grid, axis = -1)
    grid=grid[:,0,:] # array of shape (n_grid*n_grid, p): each row is composed of one element from init[0] and one element from init[1]
    return grid


# Latin Hypercube Sampling
def our_lhs(n_points,pbounds):
    '''
    Performs hypercube sampling in the range defined in pbounds. In particular samples n vectors with p components.
    Library pyDOE needed.
    Remarks: 1) the grid size is set implicitly by the choice of R, since each interval contains only one point.
             2) pbounds will be turned into a p x 2 matrix. The second number is always equal to 2.
             3) seq is a R x p matrix.
             
    '''
    dim = pbounds.shape[0]
    if type(pbounds) is dict:
        bounds = np.array([np.array(t[1]) for t in pbounds.items()])
    else:
        bounds = pbounds
    seq = lhs(dim, n_points)
    for ax in range(dim):
        m = min(bounds[ax,]) # len(bounds[ax,]) = 2 always \forall p
        M = max(bounds[ax,]) # len(bounds[ax,]) = 2 always \forall p
        for i in range(n_points):
            seq[i,ax] = m + seq[i,ax]*(M-m)
    return seq


# Implementation algorithm 2 (Frazier)

def _kg2(x, optimizer, gp, n_grid = 100, J = 300):

    
    def x_obs_to_array(d, dim):
        ''' 
        Transforms a dictionary d into an array out (used STRICTLY for x_obs)
         - d: input dictionary
         - p: number of coordinates (e.g. if we are working in R^6, then p = 6)
        '''
        out = [np.zeros(dim)] * len(d)
        for i in range(len(d)):
            # d[i] is a vector with only one element --> d[i][0] is a dictionary
            v = d[i][0].items() 
            out[i] = [item[1] for item in v]
        out = np.array(out)
        return out
    
    
    
    dim = optimizer._space.bounds.shape[0] # size of state space (e.g. if we are working on R^3, n = 3)
    x_obs_temp = np.array([[res["params"]] for res in optimizer.res]) 
    y_obs = np.array([res["target"] for res in optimizer.res])
    # x_obs needs to be np.array of size (n,dim)
    # y_obs needs to be np.array of size (n,)
    # see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    x_obs = x_obs_to_array(x_obs_temp, dim)
    
    grid = grid_construction(optimizer._space.bounds, n_grid)
                 
    temp = find_mean_max(x_obs,y_obs,gp,grid)
    mean_nstar = temp[0]

    count = 0
    
    if dim == 1:
        x = np.array([x])
        
    diff = np.zeros(len(x))
    for x_new in x:
        print(count+1)
        mean, std= gp.predict(x_new.reshape(1,-1), return_std=True)
        for j in range(J): # J = number of Monte Carlo iterations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp = find_mean_max(x_obs,y_obs,gp,grid,mean,std,x_new)
                mean_n1star = temp[0]
                diff[count] = diff[count] + (mean_n1star-mean_nstar)
        count = count + 1
    diff = diff/J
    return diff


# Implementation algorithm 4 (Frazier)

def _kg4(x, optimizer, gp, n_grid = 100, J = 300):
    '''
    Library "numdifftools" needed:
    
                                    pip install numdifftools
    REMARK: if p == 1, then x needs to be a np.float
            if p > 1, then x needs to be a np.ndarray with of the form [,]
    '''
    dim = optimizer._space.bounds.shape[0]
    
    if dim == 1:
        x = np.array(x)

    if type(x) is not np.ndarray:
        sys.exit('Error in _kg4: if p>1, x needs to be a NP.ndarray')
    if len(x.shape) > 1:
        if x.shape[0] > 1:
            sys.exit('Error in _kg4: x needs to have 1 row and p columns.')
    
    def x_obs_to_array(d, dim):
        ''' 
        Transforms a dictionary d into an array out (used STRICTLY for x_obs)
         - d: input dictionary
         - p: number of coordinates (e.g. if we are working in R^6, then p = 6)
        '''
        out = [np.zeros(dim)] * len(d)
        for i in range(len(d)):
            # d[i] is a vector with only one element --> d[i][0] is a dictionary
            v = d[i][0].items() 
            out[i] = [item[1] for item in v]
        out = np.array(out)
        return out
    
    
    def mu_x_star_hat(x_obs,y_obs,gp,x_star_hat,mean,std,i):
        '''
        see Algorithm 4:
        In our setting, we apply this by letting x_star_hat be the x^prime maximizing µ_{n+1}(x^prime; x, µ_n(x)+σ_n(x)Z), 
        and then calculating the gradient of µ_{n+1}(x_star_hat; x, µ_n(x)+σ_n(x)Z) with respect to x 
        while holding x_star_hat fixed.
        '''
        y=np.random.normal(loc=mean, scale=std)
        x_newobs=np.concatenate((x_obs, i.reshape(1,-1)), axis = 0)
        y_newobs=np.append(y_obs,y)
        mu=posterior(gp, x_newobs, y_newobs, x_star_hat)
        return np.asscalar(mu)
    
    dim = optimizer._space.bounds.shape[0] # size of state space (e.g. if we are working on R^3, n = 3) 
    x_obs_temp = np.array([[res["params"]] for res in optimizer.res]) 
    y_obs = np.array([res["target"] for res in optimizer.res])
    # x_obs needs to be np.array of size (n,p)
    # y_obs needs to be np.array of size (n,)
    # see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    x_obs = x_obs_to_array(x_obs_temp, dim)

    grid = grid_construction(optimizer._space.bounds, n_grid)
    
    temp = find_mean_max(x_obs,y_obs,gp,grid)
    mean_nstar = temp[0]

    grad_temp = []
    mean, std= gp.predict(x.reshape(1,-1), return_std=True)
    for j in range(J):
        print("j = ",j+1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp = find_mean_max(x_obs,y_obs,gp,grid,mean,std,x)
            mean_n1star = temp[0]
            x_star_hat = grid[temp[1]].reshape(1,-1)
            def func(w):
                return mu_x_star_hat(x_obs,y_obs,gp,x_star_hat,mean,std,w)
            
            grad_temp.append(nd.Gradient(func)(x))
    grad = np.zeros(dim)
    for i in grad_temp:
        grad = grad + i
    grad = grad/J
    return grad



# Implementation algorithm 3 (Frazier): Knowledge gradient 
def minimize_kg_sgd(R, T, a, pbounds, optimizer, gp):

    KG = np.zeros(R)
    init_lhs = our_lhs(R,pbounds)
    dim = pbounds.shape[0]

    if dim == 1:
        
        out = np.zeros(1)
        x0 = np.zeros((T+1,R)) 
        
        for r in range(R):
            # choose x0(r) uniformly random from A

            x0[0,r] = init_lhs[r]

            for t in range(1,T+1):
                G = _kg4(x0[t-1,r], optimizer, gp, n_grid = 50, J = 5)
                alpha = a/(a+t)
                x0[t,r] = x0[t-1,r] + alpha*G
            KG[r] = _kg2(x0[T,r], optimizer, gp, n_grid = 50, J = 5)

        out[0] = x0[T, np.argmax(KG)]
        
        return out 


    else:
        x0 = []  

        for r in range(R):
            # choose x0(r) uniformly random from A

            x0.append(init_lhs[r].reshape(1,-1)) 
            for t in range(1,T+1):
                G = _kg4(x0[(t-1)+T*(r)].reshape(1,-1), optimizer, gp, n_grid = 50, J = 5)
                alpha = a/(a+t)
                x0.append(x0[(t-1)+T*(r)] + alpha*G)      
            KG[r] = _kg2(x0[T*r].reshape(1,-1), optimizer, gp, n_grid = 50, J = 5)
    
        
        out = x0[T*np.argmax(KG)].reshape(1,-1)
        return out[0]
    
    

def acq_max(optimizer, ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10, R=1, T=5, a=4):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """
    
     
    if ac(None, optimizer, gp, y_max=0) == 'kg': 
        return minimize_kg_sgd(R, T, a, bounds, optimizer, gp)

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, optimizer, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), optimizer, gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi
        
        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi', 'kg']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, poi or kg.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, optimizer, gp, y_max):
        if self.kind == 'kg' and type(x) is not np.ndarray:
            return 'kg'
        if self.kind != 'kg' and type(x) is not np.ndarray:
            return None
        
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
       
    
    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
  
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)
    
    

def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)

