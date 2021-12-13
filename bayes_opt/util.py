
import numpy as np
import os, sys
import time

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# import sys
# # sys.path is a list of absolute path strings
# sys.path.append('/Users/anna/OneDrive/Desktop/bayesianopt seria/BayesianOptimization/Cornell_MOE_temp')

# from examples import main


# from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
# from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import PosteriorMeanMCMC
# from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import GaussianProcessLogLikelihoodMCMC as cppGaussianProcessLogLikelihoodMCMC
# from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
# from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
# from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import posterior_mean_optimization, PosteriorMean

# from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
# from moe.optimal_learning.python.geometry_utils import ClosedInterval
# from moe.optimal_learning.python.repeated_domain import RepeatedDomain
# from moe.optimal_learning.python.default_priors import DefaultPrior

# from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
# from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters as pyGradientDescentParameters
# from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer as pyGradientDescentOptimizer
# from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

# from examples import bayesian_optimization
# from examples import synthetic_functions
# from hesbo_embed import projection


# import numpy

# from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement as cppExpectedImprovement
# from moe.optimal_learning.python.cpp_wrappers.expected_improvement import multistart_expected_improvement_optimization
# from moe.optimal_learning.python.cpp_wrappers.expected_improvement_mcmc import multistart_expected_improvement_mcmc_optimization
# from moe.optimal_learning.python.cpp_wrappers.expected_improvement_mcmc import ExpectedImprovementMCMC as cppExpectedImprovementMCMC

# from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import KnowledgeGradientMCMC as cppKnowledgeGradientMCMC
# from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import multistart_knowledge_gradient_mcmc_optimization

# from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer

# # arguments for calling this script:
# # python main.py [obj_func_name] [method_name] [num_to_sample] [job_id] [hesbo_flag] [effective_dim]
# # example: python main.py Branin KG 4 1
# # you can define your own obj_function and then just change the objective_func object below, and run this script.

# argv = sys.argv[1:]
# obj_func_name = str(argv[0])
# method = str(argv[1])
# num_to_sample = int(argv[2])
# job_id = int(argv[3])
# if len(argv)>4:
#     hesbo = str(argv[4])
# else:
#     hesbo = None
    
    


# # constants
# num_func_eval = 12
# num_iteration = int(old_div(num_func_eval, num_to_sample)) + 1

# obj_func_dict = {'Branin': synthetic_functions.Branin(),
#                  'Rosenbrock': synthetic_functions.Rosenbrock(),
#                  'Hartmann3': synthetic_functions.Hartmann3(),
#                  'Levy4': synthetic_functions.Levy4(),
#                  'Hartmann6': synthetic_functions.Hartmann6(),
#                  'Ackley': synthetic_functions.Ackley()}

# objective_func = obj_func_dict[obj_func_name]

# if len(argv)>5:
#     effect_dim = int(argv[5])
# elif len(argv)>4:
#     effect_dim = int(min(6, objective_func._dim/4))

# # adjusting the test function based on the HeSBO flag
# if hesbo == 'HeSBO':
#     objective_func=projection(effect_dim, objective_func)
# elif len(argv)>4:
# 	print('WARNING: The algorithm is not using HeSBO, if you want to use HeSBO embedding, check the spelling of the input argument to be HeSBO')


# dim = int(objective_func._dim)
# num_initial_points = int(objective_func._num_init_pts)

# num_fidelity = objective_func._num_fidelity

# inner_search_domain = pythonTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
#                                                  for i in range(objective_func._search_domain.shape[0]-num_fidelity)])
# cpp_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])
# cpp_inner_search_domain = cppTensorProductDomain([ClosedInterval(objective_func._search_domain[i, 0], objective_func._search_domain[i, 1])
#                                                   for i in range(objective_func._search_domain.shape[0]-num_fidelity)])

# # get the initial data
# init_pts = np.zeros((objective_func._num_init_pts, objective_func._dim))
# init_pts[:, :objective_func._dim-objective_func._num_fidelity] = inner_search_domain.generate_uniform_random_points_in_domain(objective_func._num_init_pts)
# for pt in init_pts:
#     pt[objective_func._dim-objective_func._num_fidelity:] = np.ones(objective_func._num_fidelity)

# # observe
# derivatives = objective_func._observations
# observations = [0] + [i+1 for i in derivatives]
# init_pts_value = np.array([objective_func.evaluate(pt) for pt in init_pts])#[:, observations]
# true_value_init = np.array([objective_func.evaluate_true(pt) for pt in init_pts])#[:, observations]

# init_data = HistoricalData(dim = objective_func._dim, num_derivatives = len(derivatives))
# init_data.append_sample_points([SamplePoint(pt, [init_pts_value[num, i] for i in observations],
#                                             objective_func._sample_var) for num, pt in enumerate(init_pts)])

# # initialize the model
# prior = DefaultPrior(1+dim+len(observations), len(observations))

# # noisy = False means the underlying function being optimized is noise-free
# cpp_gp_loglikelihood = cppGaussianProcessLogLikelihoodMCMC(historical_data = init_data,
#                                                            derivatives = derivatives,
#                                                            prior = prior,
#                                                            chain_length = 1000,
#                                                            burnin_steps = 2000,
#                                                            n_hypers = 2 ** 4,
#                                                            noisy = False)
# cpp_gp_loglikelihood.train()

# py_sgd_params_ps = pyGradientDescentParameters(max_num_steps=1000,
#                                                max_num_restarts=3,
#                                                num_steps_averaged=15,
#                                                gamma=0.7,
#                                                pre_mult=1.0,
#                                                max_relative_change=0.02,
#                                                tolerance=1.0e-10)

# cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1,
#                                                  max_num_steps=6,
#                                                  max_num_restarts=1,
#                                                  num_steps_averaged=3,
#                                                  gamma=0.0,
#                                                  pre_mult=1.0,
#                                                  max_relative_change=0.1,
#                                                  tolerance=1.0e-10)

# cpp_sgd_params_kg = cppGradientDescentParameters(num_multistarts=200,
#                                                  max_num_steps=50,
#                                                  max_num_restarts=2,
#                                                  num_steps_averaged=4,
#                                                  gamma=0.7,
#                                                  pre_mult=1.0,
#                                                  max_relative_change=0.5,
#                                                  tolerance=1.0e-10)

# # minimum of the mean surface
# eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e3))
# eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
#                       (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled,
#                        cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

# test = np.zeros(eval_pts.shape[0])
# ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models, num_fidelity)
# for i, pt in enumerate(eval_pts):
#     ps.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
#     test[i] = -ps.compute_objective_function()
# report_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity))

# py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = inner_search_domain)
# ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
# report_point = multistart_optimize(ps_mean_opt, report_point, num_multistarts = 1)[0]
# report_point = report_point.ravel()
# report_point = np.concatenate((report_point, np.ones(objective_func._num_fidelity)))

# print("best so far in the initial data {0}".format(true_value_init[np.argmin(true_value_init[:,0])][0]))
# capital_so_far = 0.
# for n in range(num_iteration):
#     print(method + ", {0}th job, {1}th iteration, func={2}, q={3}".format(
#             job_id, n, obj_func_name, num_to_sample
#     ))
#     time1 = time.time()
#     if method == 'KG':
#         discrete_pts_list = []

#         discrete, _ = bayesian_optimization.gen_sample_from_qei_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_search_domain,
#                                                                 cpp_sgd_params_kg, 10, num_mc=2 ** 10)
#         for i, cpp_gp in enumerate(cpp_gp_loglikelihood.models):
#             discrete_pts_optima = np.array(discrete)

#             eval_pts = inner_search_domain.generate_uniform_random_points_in_domain(int(1e3))
#             eval_pts = np.reshape(np.append(eval_pts,
#                                             (cpp_gp.get_historical_data_copy()).points_sampled[:, :(cpp_gp_loglikelihood.dim-objective_func._num_fidelity)]),
#                                   (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim-objective_func._num_fidelity))

#             test = np.zeros(eval_pts.shape[0])
#             ps_evaluator = PosteriorMean(cpp_gp, num_fidelity)
#             for i, pt in enumerate(eval_pts):
#                 ps_evaluator.set_current_point(pt.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
#                 test[i] = -ps_evaluator.compute_objective_function()

#             initial_point = eval_pts[np.argmin(test)]

#             ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)
#             report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess = initial_point, max_num_threads = 4)

#             ps_evaluator.set_current_point(report_point.reshape((1, cpp_gp_loglikelihood.dim-objective_func._num_fidelity)))
#             if -ps_evaluator.compute_objective_function() > np.min(test):
#                 report_point = initial_point

#             discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
#                                              (discrete_pts_optima.shape[0] + 1, cpp_gp.dim-objective_func._num_fidelity))
#             discrete_pts_list.append(discrete_pts_optima)

#         ps_evaluator = PosteriorMean(cpp_gp_loglikelihood.models[0], num_fidelity)
#         ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_inner_search_domain, ps_evaluator, cpp_sgd_params_ps)
#         # KG method
#         next_points, voi = bayesian_optimization.gen_sample_from_qkg_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_gp_loglikelihood.models,
#                                                                 ps_sgd_optimizer, cpp_search_domain, num_fidelity, discrete_pts_list,
#                                                                 cpp_sgd_params_kg, num_to_sample, num_mc=2 ** 7)

def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10):
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

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
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

        if kind not in ['ucb', 'ei', 'poi','kg']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei,kg or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == 'kg':
            return self._kg(x, gp, self.xi, init=np.linspace(0, 3, 10), n=10)

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

    @staticmethod
    def _kg(x, gp, xi, init,n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a=0
            for i in range(n):
                mean, std = gp.predict(x, return_std=True)
                y=norm.pdf(x)*std+mean
                mean_n= max(gp.predict(init,return_std=False))
                gp.probe(params= {"x" : x, "y" : y},lazy=True)
                mean_n1= max(gp.predict(init,return_std=False))
                a = (mean_n - mean_n1)
                
        a = a / n
        return a 
# import numpy

# from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement as cppExpectedImprovement
# from moe.optimal_learning.python.cpp_wrappers.expected_improvement import multistart_expected_improvement_optimization
# from moe.optimal_learning.python.cpp_wrappers.expected_improvement_mcmc import multistart_expected_improvement_mcmc_optimization
# from moe.optimal_learning.python.cpp_wrappers.expected_improvement_mcmc import ExpectedImprovementMCMC as cppExpectedImprovementMCMC

# from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import KnowledgeGradientMCMC as cppKnowledgeGradientMCMC
# from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import multistart_knowledge_gradient_mcmc_optimization

# from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
   
  # @staticmethod
  #     def _kg(x, gp, y_max, xi):
  #         cpp_kg_evaluator = cppKnowledgeGradientMCMC(gaussian_process_mcmc = cpp_gp_mcmc, gaussian_process_list=cpp_gp_list,
  #                                                     num_fidelity = num_fidelity, inner_optimizer = inner_optimizer, discrete_pts_list=discrete_pts_list,
  #                                                     num_to_sample = num_to_sample, num_mc_iterations=int(num_mc))
             
  #         return cpp_kg_evaluator
     

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
