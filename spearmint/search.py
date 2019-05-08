import os
import time
import signal
import shutil
import collections
import numpy as np
import pandas as pd
from spearmint.ExperimentGrid import GridMap
from spearmint.chooser import GPEIOptChooser as module
from sklearn.model_selection import cross_val_score

grid_seed = 1 # the number of initial points to skip
grid_size = 1000

def set_timeout(func):
    def handle(signum, frame): 
        raise RuntimeError

    def to_do(*args, **kwargs):
        try:
            signal.signal(signal.SIGALRM, handle)  
            signal.alarm(args[0].time_out)   
            r = func(*args, **kwargs)
            signal.alarm(0)   
            return r
        except RuntimeError as e:
            print("Time out!")
    return to_do
    
class GPEISklearn():
    """ 
     Sklearn Hyperparameter optimization interface based on Gaussian Process - Expected Improvement (Bayesian Optimization). 
    
        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.
        cv : cross-validation method, an sklearn object.
            e.g., `StratifiedKFold` and KFold` is used.
        para_space : dict or list of dictionaries
            It has three types:
            Continuous: 
                Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
                `Wrapper`, a callable function for wrapping the values.  
            Integer:
                Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
            Categorical:
                Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).
        max_runs: int, optional, default = 100
            The maximum number of trials to be evaluated. When this values is reached, 
            then the algorithm will stop. 
        time_out: float, optional, default = 10
            The time out threshold (in seconds) for generating the next run. 
        scoring : string, callable, list/tuple, dict or None, optional, default = None
            A sklearn type scoring function. 
            If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.
        refit : boolean, or string, optional, default=True
            It controls whether to refit an estimator using the best found parameters on the whole dataset.
        verbose : boolean, optional, default = False
            It controls whether the searching history will be printed. 


        Examples
        --------
        >>> from sklearn import svm
        >>> from pySeqUD.search import GPEISklearn
        >>> from sklearn.model_selection import KFold
        >>> iris = datasets.load_iris()
        >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'wrapper': np.exp2}, 
                   'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'wrapper': np.exp2}}
        >>> estimator = svm.SVC()
        >>> cv = KFold(n_splits=5, random_state=1, shuffle=True)
        >>> clf = GPEISklearn(estimator, cv, ParaSpace, refit = True, verbose = True)
        >>> clf.fit(iris.data, iris.target)

        Attributes
        ----------
        best_score_ : float
            The best average cv score among the evaluated trials.  

        best_params_ : dict
            Parameters that reaches `best_score_`.

        best_estimator_: 
            The estimator refitted based on the `best_params_`. 
            Not available if `refit=False`.

        search_time_consumed_: float
            Seconds used for whole searching procedure.

        refit_time_: float
            Seconds used for refitting the best model on the whole dataset.
            Not available if `refit=False`.
    """    

    def __init__(self, estimator, cv, para_space, max_runs = 100, time_out = 10,
                 scoring=None, refit=False, rand_seed = 0, verbose=False):

        self.estimator = estimator        
        self.cv = cv
        
        self.para_space = para_space
        self.rand_seed = rand_seed
        self.max_runs = max_runs

        self.time_out = time_out
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.factor_number = len(self.para_space)
        self.para_names = list(self.para_space.keys())
        
    def _summary(self):
        """
        This function summarizes the evaluation results and makes records. 
        
        """

        self.best_index_ = self.logs.loc[:,"score"].idxmax()
        self.best_params_ = {self.logs.loc[:,self.para_names].columns[j]:\
                             self.logs.loc[:,self.para_names].iloc[self.best_index_,j] 
                              for j in range(self.logs.loc[:,self.para_names].shape[1])}
        
        self.best_score_ = self.logs.loc[:,"score"].iloc[self.best_index_]

        if self.verbose:
            para_print_list = []
            for items in self.para_names:
                if self.para_space[items]["Type"]=="continuous":
                    pr_string = items + " = %.5f" % self.best_params_[items]
                if self.para_space[items]["Type"]=="integer": 
                    pr_string = items + " = %d" % self.best_params_[items]
                if self.para_space[items]["Type"]=="categorical":
                    pr_string = items + " = %s" % self.best_params_[items]
                para_print_list.append(pr_string)
            self.para_print = ' '.join(para_print_list)
            print("Search completed in %.2f seconds, with best score: %.5f."%(self.search_time_consumed_, self.best_score_))
            print("Best parameters: %s"%self.para_print)

    def _para_mapping(self):
        """
        This function configures different hyperparameter types of spearmint. 
        
        """

        self.variables = {}
        for items, values in self.para_space.items():
            if (values['Type']=="continuous"):
                self.variables[items] =  collections.OrderedDict({'name': items, 
                                 'type':'float',
                                 'min': values['Range'][0],
                                 'max': values['Range'][1],
                                 'size': 1})
            elif (values['Type']=="integer"):
                self.variables[items] = collections.OrderedDict({'name': items, 
                                 'type':'int',
                                 'min': min(values['Mapping']),
                                 'max': max(values['Mapping']),
                                 'size': 1})
            elif (values['Type']=="categorical"):
                self.variables[items] = collections.OrderedDict({'name': items, 
                                 'type':'enum',
                                 'options': values['Mapping'],
                                 'size': 1}) 

    @set_timeout  
    def spmint_opt(self, params, values, variables, file_dir):
        """
        Interface for generating next run based on spearmint. 
        
        """

        chooser = module.init(file_dir, "mcmc_iters=10")
        vkeys = [k for k in variables]

        #vkeys.sort()
        gmap = GridMap([variables[k] for k in vkeys], grid_size)
        candidates = gmap.hypercube_grid(grid_size, grid_seed+params.shape[0])
        if (params.shape[0] > 0):
            grid = np.vstack((params, candidates))
        else:
            grid = candidates
        grid = np.asarray(grid)
        grid_idx = np.hstack((np.zeros(params.shape[0]),
                              np.ones(candidates.shape[0])))
        job_id = chooser.next(grid, np.squeeze(values), [],
                              np.nonzero(grid_idx == 1)[0],
                              np.nonzero(grid_idx == 2)[0],
                              np.nonzero(grid_idx == 0)[0])

        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
        else:
            candidate = grid[job_id,:]
        next_params = gmap.unit_to_list(candidate)
        next_params = pd.DataFrame(np.array([next_params]), columns = self.para_names)
        next_params_dict = {}
        for items, values in self.para_space.items():
            if (values['Type']=="continuous"):
                next_params_dict[items] = values['Wrapper'](float(next_params[items]))
            elif (values['Type']=="integer"):
                next_params_dict[items] = int(next_params[items]) 
            elif (values['Type']=="categorical"):
                next_params_dict[items] = next_params[items][0]
        return candidate, next_params_dict
    
    def spearmint_run(self, obj_func_):
        """
        Main loop for searching the best hyperparameters. 
        
        """
        params = []
        scores = []
        param_unit = []
        file_dir = "./temp/" + str(time.time()) + str(np.random.rand(1)[0]) + "/"
        os.makedirs(file_dir)
        np.random.seed(self.rand_seed)
        for i in range(self.max_runs):
            try:
                candidate, next_params = self.spmint_opt(np.array(param_unit), -np.array(scores), self.variables, file_dir)
            except:
                print("Early Stop!")
                break
                
            score = obj_func_(next_params)
            param_unit.append(candidate)
            params.append(next_params)
            scores.append(score)
            if self.verbose:
                para_print_list = []
                for items in self.para_names:
                    if self.para_space[items]["Type"]=="continuous":
                        pr_string = items + " = %.5f" % next_params[items]
                    if self.para_space[items]["Type"]=="integer": 
                        pr_string = items + " = %d" % next_params[items]
                    if self.para_space[items]["Type"]=="categorical":
                        pr_string = items + " = %s" % next_params[items]
                    para_print_list.append(pr_string)
                para_print = ' '.join(para_print_list)
                print("Iteration (%d/%d) with score: %.5f."%(i+1, self.max_runs, score))

        scores = np.array(scores).reshape([-1,1])
        self.logs = pd.concat([pd.DataFrame(params, columns = self.para_names), 
                               pd.DataFrame(scores, columns = ["score"])], axis = 1)
        shutil.rmtree(file_dir)
            
    def fit(self, x, y = None):
        """
        Run fit with all sets of parameters.
        Parameters
        ----------
        x : array, shape = [n_samples, n_features], input variales
        y : array, shape = [n_samples] or [n_samples, n_output], optional target variable
        """
        def obj_func_wrapper(parameters):
            self.estimator.set_params(**parameters)
            out = cross_val_score(self.estimator, x, y, cv = self.cv, scoring = self.scoring)
            score = np.mean(out)
            return score

        
        search_start_time = time.time()
        self._para_mapping()
        self.spearmint_run(obj_func_wrapper)
        search_end_time = time.time()
        self.search_time_consumed_ = search_start_time - search_end_time

        self._summary()
        
        if self.refit:
            self.best_estimator_ = self.estimator.set_params(**self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(x, y)
            else:
                self.best_estimator_.fit(x)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time