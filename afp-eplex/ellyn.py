# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""

import argparse
# from ._version import __version__

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_error, explained_variance_score, accuracy_score
import numpy as np
import pandas as pd
import warnings
import copy
import itertools as it
import pdb
import ellen.lib.elgp as elgp
#from update_checker import update_check
from DistanceClassifier import DistanceClassifier
from functools import wraps
import inspect
import math

def initializer(fun):
    """automatically assigns attributes to ellyn class upon instantiation"""
    names, varargs, keywords, defaults = inspect.getargspec(fun)
    @wraps(fun)
    def wrapper(self, *args, **kargs):
       if type(kargs.items()) is list:
           for name, arg in zip(names[1:], args) + kargs.items():
               setattr(self, name, arg)
       else:
           for name, arg in zip(names[1:], args) | kargs.items():
               setattr(self, name, arg)
       fun(self, *args, **kargs)
    return wrapper

class ellyn(BaseEstimator):
    """ellyn uses GP to build its models.

    It uses a stack-based, syntax-free, linear genome for constructing candidate equations.
    It is built to include different evolutionary methods for system identification
    adapted from literature. The options include normal tournament selection,
    deterministic crowding, (epsilon) lexicase selection, and age-pareto fitness selection.

    All algorithm choices are accessible from the command line.

    """
    update_checked = False
    @initializer
    def __init__(self, g=100, popsize=500, limit_evals=False, max_evals=0,
                 selection='tournament', classification=False, islands=True,
                 num_islands=None,fit_type=None, verbosity=0, random_state=0,
                 class_m4gp=False,scoring_function=None, print_log=False,
                 print_archive=False,print_data=False, print_db=False,
                 class_bool=False, max_len=None, island_gens=50,
                 print_every_pop=None, FE_pop_size=None, lex_eps_global=None,
                 rt_cross=None, ERC_ints=None, numERC=None, train_pct=None,
                 eHC_on=None, PS_sel=None, lex_eps_static=None,
                 lex_eps_semidynamic=None, lex_eps_dynamic=None,
                 lex_eps_dynamic_rand=None,lex_eps_dynamic_madcap=None,
                 lex_pool=None, AR_nka=None, lex_meta=None, train=None,
                 print_homology=None, max_len_init=None, prto_arch_size=None,
                 cvals=None, stop_condition=None, stop_threshold=None,
                 FE_rank=None, eHC_its=None, lex_eps_error_mad=True,
                 ERC=None, erc_ints=None, AR_na=None, rt_mut=None,
                 pop_restart=None, seeds=None, tourn_size=None, prto_arch_on=None,
                 FE_ind_size=None, lex_eps_target_mad=None, maxERC=None,
                 resultspath=None, AR=None, rt_rep=None, estimate_fitness=None,
                 pHC_on=None, INPUT_FILE=None, FE_train_size=None,
                 DISABLE_UPDATE_CHECK=False, AR_lookahead=None, pop_restart_path=None,
                 INPUT_SEPARATOR=None, AR_nkb=None, num_log_pts=None,
                 FE_train_gens=None, AR_nb=None, init_trees=None,
                 print_novelty=None, eHC_slim=None, elitism=None,
                 print_genome=None, pHC_its=None, shuffle_data=None, class_prune=None,
                 eHC_init=None, init_validate_on=None, minERC=None,
                 ops=None, ops_w=None, eHC_mut=None, min_len=None, return_pop=False,
                 savename=None,lexage=None,SGD=None,learning_rate=None):
                # sets up GP.
        self.classification = classification
        if scoring_function is None:
            if fit_type:
                self.scoring_function = {'MAE':mean_absolute_error,
                                         'MSE':mean_squared_error,
                                         'R2':r2_score,
                                         'VAF':explained_variance_score,
                                         'combo':mean_absolute_error,
                                         'F1':accuracy_score,
                                         'F1W':accuracy_score}[fit_type]
            elif classification:
                # self.fit_type = 'MAE'
                self.scoring_function = accuracy_score
            else:
                self.scoring_function = mean_squared_error
        else:
            self.scoring_function = scoring_function

        self.random_state = random_state
        self.selection  = selection
        self.prto_arch_on = prto_arch_on
        self.AR = AR
        self.verbosity = verbosity
        self.best_estimator_ = []
        self.hof = []
        self.return_pop = return_pop
        #convert m4gp argument to m3gp used in ellenGP
        if classification and not class_m4gp and not class_bool:
            #default to M4GP in the case no classifier specified
            self.class_m4gp = True
        elif not classification:
            self.class_m4gp = False

        # set op_list
        if ops:
            self.op_list = ops.split(',')
        if ops_w:
            self.weight_ops_on = True
            self.op_weight = [float(x) for x in ops_w.split(',')]

        if lex_meta:
            self.lex_metacases = lex_meta.split(',')

    def fit(self, features, labels):
        """Fit model to data"""
        # set sel number from selection
        self.sel = {'tournament': 1,'dc':2,'lexicase': 3,'epsilon_lexicase': 3,
                    'afp': 4,'rand': 5,None: 1}[self.selection]

        if self.selection=='epsilon_lexicase':
            self.lex_eps_error_mad = True

        np.random.seed(self.random_state)
        # get parameters
        params = dict(self.__dict__)

        for k in list(params.keys()):
            if params[k] is None:
                del params[k]

        if self.prto_arch_on:
            # split data into training and internal validation for choosing
            # final model
            if self.AR:
                # don't shuffle the rows of ordered data
                train_i = np.arange(round(features.shape[0]*.75))
                val_i = np.arange(round(features.shape[0]*.75),features.shape[0])
            else:
                stratify=None
                # if classification, make the train/test split even across class
                if self.classification:
                    stratify = labels
                    train_i, val_i = train_test_split(np.arange(features.shape[0]),
                                                stratify=stratify,
                                                train_size=0.75,
                                                test_size=0.25,
                                                random_state=self.random_state)
                    features = features[list(train_i)+list(val_i)]
                    labels = labels[list(train_i)+list(val_i)]
                params['train'] = True
                params['train_pct'] = 0.75

        result = []
        if self.verbosity>1:
            print(10*'=','params',10*'=',sep='\n')
            for key,item in params.items():
                print(key,':',item)
        # run ellenGP
        elgp.runEllenGP(params,
                        np.asarray(features,dtype=np.float32,order='C'),
                        np.asarray(labels,dtype=np.float32,order='C'),
                        result)
        # print("best program:",self.best_estimator_)
        if self.AR: # set defaults
            if not self.AR_na: self.AR_na = 0
            if not self.AR_nka: self.AR_nka = 1
            if not self.AR_nb: self.AR_nb = 0
            if not self.AR_nkb: self.AR_nkb = 0

        if self.prto_arch_on or self.return_pop:
            # a set of models is returned. choose the best
            if self.verbosity>0: print('Evaluating archive set...')
            # unpack results into model, train and test fitness
            self.hof = [r[0] for r in result]
            self.fit_t = [r[1] for r in result]
            self.fit_v = [r[2] for r in result]
            # best estimator has lowest internal validation score
            self.best_estimator_ = self.hof[np.argmin(self.fit_v)]
        else:
            # one model is returned
            self.best_estimator_ = result

        # if M4GP is used, call Distance Classifier and fit it to the best
        if self.class_m4gp:
            if self.verbosity > 0: print("Storing DistanceClassifier...")
            self.DC = DistanceClassifier()
            self.DC.fit(self._out(self.best_estimator_,features),labels)

        ####     
        # print final models
        if self.verbosity>0:
             print("final model(s):")
             if self.prto_arch_on or self.return_pop:
                 for m in self.hof[::-1]:
                     if m is self.best_estimator_:
                         print('[best]',self.stack_2_eqn(m),sep='\t')
                     else:
                         print('',self.stack_2_eqn(m),sep='\t')
             else:
                 print('best model:',self.stack_2_eqn(self.best_estimator_))


    def predict(self, testing_features,ic=None):
        """predict on a holdout data set."""
        # print("best_inds:",self._best_inds)
        # print("best estimator size:",self.best_estimator_.coef_.shape)
        # tmp = self._out(self.best_estimator_,testing_features)
        #
        if self.class_m4gp:
            return self.DC.predict(self._out(self.best_estimator_,testing_features))
        else:
            return self._out(self.best_estimator_,testing_features,ic)

    def fit_predict(self, features, labels):
        """Convenience function that fits a pipeline then predicts on the provided features

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        labels: array-like {n_samples}
            List of class labels for prediction

        Returns
        ----------
        array-like: {n_samples}
            Predicted labels for the provided features

        """
        self.fit(features, labels)
        return self.predict(features)

    def score(self, testing_features, testing_labels,ic=None):
        """estimates accuracy on testing set"""
        # print("test features shape:",testing_features.shape)
        # print("testing labels shape:",testing_labels.shape)
        yhat = self.predict(testing_features,ic)
        return self.scoring_function(testing_labels,yhat)

    def export(self, output_file_name):
        """does nothing currently"""

    def _eval(self,n, features, stack_float, stack_bool, y= None):
        """evaluation function for best estimator"""
        eval_dict = {
        # float operations
            '+': lambda n,features,stack_float,stack_bool: stack_float.pop() + stack_float.pop(),
            '-': lambda n,features,stack_float,stack_bool: stack_float.pop() - stack_float.pop(),
            '*': lambda n,features,stack_float,stack_bool: stack_float.pop() * stack_float.pop(),
            '/': lambda n,features,stack_float,stack_bool: self.divs(stack_float.pop(),stack_float.pop()),
            's': lambda n,features,stack_float,stack_bool: np.sin(stack_float.pop()),
            'c': lambda n,features,stack_float,stack_bool: np.cos(stack_float.pop()),
            'e': lambda n,features,stack_float,stack_bool: np.exp(stack_float.pop()),
            'l': lambda n,features,stack_float,stack_bool: self.logs(np.asarray(stack_float.pop())),#np.log(np.abs(stack_float.pop())),
            'x':  lambda n,features,stack_float,stack_bool: features[:,n[2]],
            'k': lambda n,features,stack_float,stack_bool: np.ones(features.shape[0])*n[2],
            '2': lambda n,features,stack_float,stack_bool: stack_float.pop()**2,
            '3': lambda n,features,stack_float,stack_bool: stack_float.pop()**3,
            'q': lambda n,features,stack_float,stack_bool: np.sqrt(np.abs(stack_float.pop())),
            'xd': lambda n,features,stack_float,stack_bool: features[-1-n[3],n[2]],
            'kd': lambda n,features,stack_float,stack_bool: n[2],
        # bool operations
            '!': lambda n,features,stack_float,stack_bool: not stack_bool.pop(),
            '&': lambda n,features,stack_float,stack_bool: stack_bool.pop() and stack_bool.pop(),
            '|': lambda n,features,stack_float,stack_bool: stack_bool.pop() or stack_bool.pop(),
            '==': lambda n,features,stack_float,stack_bool: stack_bool.pop() == stack_bool.pop(),
            '>': lambda n,features,stack_float,stack_bool: stack_float.pop() > stack_float.pop(),
            '<': lambda n,features,stack_float,stack_bool: stack_float.pop() < stack_float.pop(),
            '}': lambda n,features,stack_float,stack_bool: stack_float.pop() >= stack_float.pop(),
            '{': lambda n,features,stack_float,stack_bool: stack_float.pop() <= stack_float.pop(),
            # '>_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() > stack_bool.pop(),
            # '<_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() < stack_bool.pop(),
            # '>=_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() >= stack_bool.pop(),
            # '<=_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() <= stack_bool.pop(),
        }

        def safe(x):
            """removes nans and infs from outputs."""
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
            return x

        np.seterr(all='ignore')
        if len(stack_float) >= n[1]:
            if n[0] == 'y': # return auto-regressive variable
                if len(y)>=n[2]:
                    stack_float.append(y[-n[2]])
                else:
                    stack_float.append(0.0)
            else:
                stack_float.append(safe(eval_dict[n[0]](n,features,stack_float,stack_bool)))
            try:
                if np.any(np.isnan(stack_float[-1])) or np.any(np.isinf(stack_float[-1])):
                    print("problem operator:",n)
            except Exception:
                raise(Exception)

    def _out(self,I,features,ic=None):
        """computes the output for individual I"""
        stack_float = []
        stack_bool = []
        # print("stack:",I.stack)
        # evaulate stack over rows of features,labels
        # pdb.set_trace()
        if self.AR:

            delay = self.AR_nb+self.AR_nkb+1

            if ic is not None:
                tmp_features = np.vstack([ic['features'],features])
            else:
                tmp_features = np.vstack([np.zeros((self.AR_nb,features.shape[1])),features])
            #for autoregressive models, need to evaluate each sample in a loop,
            # setting delayed outputs as you go
            y = np.zeros(features.shape[0])
            # evaluate models sample by sample
            for i,f in enumerate([tmp_features[j-delay:j] for j in np.arange(features.shape[0])+delay]):
                stack_float=[]
                stack_bool=[]

                # use initial condition if one exists
                if ic is not None:
                    # pdb.set_trace()
                    tmpy = np.hstack((ic['labels'],y[:i]))
                else:
                    tmpy = y[:i]

                for n in I:
                    if n[0]=='x': pdb.set_trace()
                    if n[0]=='k':
                        n = ('kd',n[1],n[2])
                    self._eval(n,f,stack_float,stack_bool,tmpy)
                    # pdb.set_trace()
                try:
                    y[i] = stack_float[-1]
                except:
                    pdb.set_trace()
        else: # normal vectorized evaluation over all rows / samples
            for n in I:
                self._eval(n,features,stack_float,stack_bool)
                # print("stack_float:",stack_float)
        # pdb.set_trace()
        # if y.shape[0] != features.shape[0]:

        if self.class_m4gp:
            return np.asarray(stack_float).transpose()
        elif self.AR:
            #
            return np.array(y)
        else:
            return stack_float[-1]


    def stacks_2_eqns(self,stacks):
        """returns equation strings from stacks"""
        if stacks:
            return list(map(lambda p: self.stack_2_eqn(p), stacks))
        else:
            return []

    def stack_2_eqn(self,p):
        """returns equation string for program stack"""
        stack_eqn = []
        if p: # if stack is not empty
            for n in p:
                self.eval_eqn(n,stack_eqn)
        if self.class_m4gp:
            return stack_eqn
        else:
            return stack_eqn[-1]
        return []

    def eval_eqn(self,n,stack_eqn):
        if n[0] == '<':
            pdb.set_trace()
        if len(stack_eqn) >= n[1]:
            stack_eqn.append(eqn_dict[n[0]](n,stack_eqn))

    # def delay_feature(self,feature,delay):
    #     """returns delayed feature value for auto-regressive models"""
    #     # ar_feat = np.vstack((np.zeros(delay+self.AR_nkb), np.array([feature[j-(delay+self.AR_nkb)] for j in feature if j >= delay + self.AR_nkb]) ))
    #     pdb.set_trace()
    #     if len(feature)>=delay:
    #         return feature[-delay]
    #     else:
    #         return 0.0

    def divs(self,x,y):
        """safe division"""
        # pdb.set_trace()

        if type(x) == np.ndarray:
            tmp = np.ones(y.shape)
            nonzero_y = np.abs(y) >= 0.000001
            # print("nonzero_y.sum:", np.sum(nonzero_y))
            tmp[nonzero_y] = x[nonzero_y]/y[nonzero_y]
            return tmp
        else:
            # pdb.set_trace()
            if abs(y) >= 0.000001:
                return x/y
            else:
                return 1


    def logs(self,x):
        """safe log"""
        tmp = np.zeros(x.shape)
        nonzero_x = np.abs(x) >= 0.000001
        tmp[nonzero_x] = np.log(np.abs(x[nonzero_x]))
        return tmp

    def plot_archive(self):
        """plots pareto archive in terms of model accuracy and complexity"""
        if not self.hof:
            raise(ValueError,"no archive to print")
        else:
            import matplotlib.pyplot as plt

            f_t = np.array(self.fit_t)[::-1]
            f_v = np.array(self.fit_v)[::-1]
            m_v = self.hof[::-1]

            # lines
            plt.plot(f_t,np.arange(len(f_t)),'b')
            plt.plot(f_v,np.arange(len(f_v)),'r')
            plt.legend()

            for i,(m,f1,f2) in enumerate(zip(m_v,f_t,f_v)):
                plt.plot(f1,i,'sb',label='Train')
                plt.plot(f2,i,'xr',label='Validation')
                plt.text((min(f_v))*0.9,i,self.stack_2_eqn(m))
                if f2 == max(f_v):
                    plt.plot(f2,i,'ko',markersize=4)
            plt.ylabel('Complexity')
            plt.gca().set_yticklabels('')
            plt.xlabel('Score')
            xmin = min([min(f_t),min(f_v)])
            xmax = max([max(f_t),max(f_v)])
            plt.xlim(xmin*.8,xmax*1.2)
            plt.ylim(-1,len(m_v)+1)

            return plt.gcf()

    def output_archive(self):
        """prints pareto archive in terms of model accuracy and complexity
        format:
            model\tcomplexity\ttrain\ttest\n
            x1\t0.05\t1\n
        """
        out = 'model\tcomplexity\ttrain\ttest\n'

        if not self.hof:
            raise(ValueError,"no archive to print")
            return 0
           
        f_t = np.array(self.fit_t)[::-1]
        f_v = np.array(self.fit_v)[::-1]
        m_v = self.hof[::-1]

        for i,(m,f1,f2) in enumerate(zip(m_v,f_t,f_v)):
            out += '\t'.join([str(self.stack_2_eqn(m)),
                             str(i),
                             str(f1),
                             str(f2)])+'\n'

        return out

# equation conversion
eqn_dict = {
    '+': lambda n,stack_eqn: '(' + stack_eqn.pop() + '+' + stack_eqn.pop() + ')',
    '-': lambda n,stack_eqn: '(' + stack_eqn.pop() + '-' + stack_eqn.pop()+ ')',
    '*': lambda n,stack_eqn: '(' + stack_eqn.pop() + '*' + stack_eqn.pop()+ ')',
    '/': lambda n,stack_eqn: '(' + stack_eqn.pop() + '/' + stack_eqn.pop()+ ')',
    's': lambda n,stack_eqn: 'sin(' + stack_eqn.pop() + ')',
    'c': lambda n,stack_eqn: 'cos(' + stack_eqn.pop() + ')',
    'e': lambda n,stack_eqn: 'exp(' + stack_eqn.pop() + ')',
    'l': lambda n,stack_eqn: 'log(' + stack_eqn.pop() + ')',
    '2': lambda n,stack_eqn: '(' + stack_eqn.pop() + '^2)',
    '3': lambda n,stack_eqn: '(' + stack_eqn.pop() + '^3)',
    'q': lambda n,stack_eqn: 'sqrt(|' + stack_eqn.pop() + '|)',
    # 'rbf': lambda n,stack_eqn: 'exp(-||' + stack_eqn.pop()-stack_eqn.pop() '||^2/2)',
    'x':  lambda n,stack_eqn: 'x_' + str(n[2]),
    'k': lambda n,stack_eqn: str(round(n[2],3)),
    'y': lambda n,stack_eqn: 'y_{t-' + str(n[2]) + '}',
    'xd': lambda n,stack_eqn: 'x_' + str(n[2]) + '_{t-' + str(n[3]) + '}'
}



def positive_integer(value):
    """Ensures that the provided value is a positive integer; throws an exception otherwise

    Parameters
    ----------
    value: int
        The number to evaluate

    Returns
    -------
    value: int
        Returns a positive integer
    """
    try:
        value = int(value)
    except Exception:
        raise argparse.ArgumentTypeError('Invalid int value: \'{}\''.format(value))
    if value < 0:
        raise argparse.ArgumentTypeError('Invalid positive int value: \'{}\''.format(value))
    return value

def float_range(value):
    """Ensures that the provided value is a float integer in the range (0., 1.); throws an exception otherwise

    Parameters
    ----------
    value: float
        The number to evaluate

    Returns
    -------
    value: float
        Returns a float in the range (0., 1.)
    """
    try:
        value = float(value)
    except:
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    return value

# main functions
def main():
    """Main function that is called when ellyn is run on the command line"""
    parser = argparse.ArgumentParser(description='A genetic programming '
                                                 'system for regression and classification.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to run ellyn on; ensure that the target/label column is labeled as "label".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default=None,
                        type=str, help='Character used to separate columns in the input file.')

# GP Runtime Options
    parser.add_argument('-g', action='store', dest='g', default=None,
                        type=positive_integer, help='Number of generations to run ellyn.')

    parser.add_argument('-p', action='store', dest='popsize', default=None,
                        type=positive_integer, help='Number of individuals in the GP population.')

    parser.add_argument('--limit_evals', action='store_true', dest='limit_evals', default=None,
                    help='Limit evaluations instead of generations.')

    parser.add_argument('-me', action='store', dest='max_evals', default=None,
                        type=float_range, help='Max point evaluations.')

    parser.add_argument('-sel', action='store', dest='selection', default='tournament', choices = ['tournament','dc','lexicase','epsilon_lexicase','afp','rand'],
                        type=str, help='Selection method (Default: tournament)')

    parser.add_argument('-PS_sel', action='store', dest='PS_sel', default=None, choices = [1,2,3,4,5],
                        type=str, help='objectives for pareto survival. 1: age + fitness; 2: age+fitness+generality; 3: age+fitness+complexity; 4: class fitnesses (classification ONLY); 5: class fitnesses+ age (classification ONLY)')

    parser.add_argument('-tourn_size', action='store', dest='tourn_size', default=None,
                        type=positive_integer, help='Tournament size for tournament selection (Default: 2)')

    parser.add_argument('-mr', action='store', dest='rt_mut', default=None,
                        type=float_range, help='GP mutation rate in the range [0.0, 1.0].')

    parser.add_argument('-xr', action='store', dest='rt_cross', default=None,
                        type=float_range, help='GP crossover rate in the range [0.0, 1.0].')

    parser.add_argument('-rr', action='store', dest='rt_rep', default=None,
                        type=float_range, help='GP reproduction rate in the range [0.0, 1.0].')

    parser.add_argument('--elitism', action='store_true', dest='elitism', default=None,
                help='Flag to force survival of best individual in GP population.')

    parser.add_argument('--init_validate', action='store_true', dest='init_validate_on', default=None,
                help='Flag to guarantee initial population outputs are valid (no NaNs or Infs).')

    parser.add_argument('--no_stop', action='store_false', dest='stop_condition', default=None,
                    help='Flag to keep running even though fitness < 1e-6 has been reached.')
    parser.add_argument('-stop_threshold', action='store', dest='stop_threshold', default=None,
                    type=float, help='Fitness theshold for stopping execution.')
# Data Options
    parser.add_argument('--validate', action='store_true', dest='train', default=None,
                help='Flag to split data into training and validation sets.')

    parser.add_argument('-train_split', action='store', dest='train_pct', default=None, type = float_range,
                help='Fraction of data to use for training in [0,1]. Only used when --validate flag is present.')

    parser.add_argument('--shuffle', action='store_true', dest='shuffle_data', default=None,
                help='Flag to shuffle data samples.')

    parser.add_argument('--restart', action='store_true', dest='pop_restart_path', default=None,
                help='Flag to restart from previous population.')

    parser.add_argument('-pop', action='store_true', dest='pop_restart', default=None,
                help='Population to use in restart. Only used when --restart flag is present.')

    parser.add_argument('--AR', action='store_true', dest='AR', default=None,
                    help='Flag to use auto-regressive variables.')

    parser.add_argument('--AR_lookahead', action='store_true', dest='AR_lookahead', default=None,
                    help='Flag to only estimate on step ahead of data when evaluating candidate models.')

    parser.add_argument('-AR_na', action='store', dest='AR_na', default=None,
                    type=positive_integer, help='Order of auto-regressive output (y).')
    parser.add_argument('-AR_nb', action='store', dest='AR_nb', default=None,
                    type=positive_integer, help='Order of auto-regressive inputs (x).')
    parser.add_argument('-AR_nka', action='store', dest='AR_nka', default=None,
                    type=positive_integer, help='AR state (output) delay.')
    parser.add_argument('-AR_nkb', action='store', dest='AR_nkb', default=None,
                    type=positive_integer, help='AR input delay.')

# Results and Printing Options
    parser.add_argument('-op', action='store', dest='resultspath', default=None,
                        type=str, help='Path where results will be saved.')
    parser.add_argument('-on', action='store', dest='savename', default=None,
                        type=str, help='Name of the file where results will be saved.')
    parser.add_argument('--print_log', action='store_true', dest='print_log', default=False,
                    help='Flag to print log to terminal.')

    parser.add_argument('--print_every_pop', action='store_true', dest='print_every_pop', default=None,
                    help='Flag to seed initial GP population with components of the ML model.')

    parser.add_argument('--print_genome', action='store_true', dest='print_genome', default=None,
                    help='Flag to prints genome for visualization.')

    parser.add_argument('--print_novelty', action='store_true', dest='print_novelty', default=None,
                    help='Flag to print number of unique output vectors.')

    parser.add_argument('--print_homology', action='store_true', dest='print_homology', default=None,
                    help='Flag to print genetic homology in programs.')

    parser.add_argument('--print_db', action='store_true', dest='print_db', default=None,
                    help='Flag to print individuals for graph database analysis.')

    parser.add_argument('-num_log_pts', action='store', dest='num_log_pts', default=None,
                        type=positive_integer, help='number of log points to print (0 means print each generation)')

# Classification Options

    parser.add_argument('--class', action='store_true', dest='classification', default=None,
                    help='Flag to define a classification problem instead of regression (the default).')

    parser.add_argument('--class_bool', action='store_true', dest='class_bool', default=None,
                    help='Flag to interpret class labels as bit-string conversion of boolean stack output.')

    parser.add_argument('--class_m4gp', action='store_true', dest='class_m4gp', default=False,
                    help='Flag to use the M4GP algorithm.')

    parser.add_argument('--class_prune', action='store_true', dest='class_prune', default=None,
                        help='Flag to prune the dimensions of the best individual each generation.')

# Terminal and Operator Options
    parser.add_argument('-ops', action='store', dest='ops', default=None,
                    type=str, help='Operator list separated by commas (no spaces!). Default: +,-,*,/,n,v. available operators: n v + - * / sin cos log exp sqrt = ! < <= > >= if-then if-then-else & |')

    parser.add_argument('-ops_w', action='store', dest='ops_w', default=None,
                    help='Operator weights for each element in operator list, separated by commas (no spaces!). If not specified all operators have the same weights.')

    parser.add_argument('-constants', nargs = '*', action='store', dest='cvals', default=None,
                        type=float, help='Seed GP initialization with constant values.')

    parser.add_argument('-seeds', nargs = '*', action='store', dest='seeds', default=None,
                        type=str, help='Seed GP initialization with partial solutions, e.g. (x+y). Each partial solution must be enclosed in parentheses.')

    parser.add_argument('-min_len', action='store', dest='min_len', default=None,
                        type=positive_integer, help='Minimum length of GP programs.')

    parser.add_argument('-max_len', action='store', dest='max_len', default=None,
                        type=positive_integer, help='Maximum number of nodes in GP programs.')

    parser.add_argument('-max_len_init', action='store', dest='max_len_init', default=None,
                        type=positive_integer, help='Maximum number of nodes in initialized GP programs.')

    parser.add_argument('--erc_ints', action='store_true', dest='erc_ints', default=None,
                help='Flag to use integer instead of floating point ERCs.')

    parser.add_argument('--no_erc', action='store_false', dest='ERC', default=None,
                help='Flag to turn of ERCs. Useful if you are specifying constant values and don''t want random ones.')

    parser.add_argument('-min_erc', action='store', dest='minERC', default=None,
                    help='Minimum ERC value.')

    parser.add_argument('-max_erc', action='store', dest='maxERC', default=None, type = int,
                    help='Maximum ERC value.')

    parser.add_argument('-num_erc', action='store', dest='numERC', default=None, type = int,
                    help='Number of ERCs to use.')

    parser.add_argument('--trees', action='store_true', dest='init_trees', default=None,
                    help='Flag to initialize genotypes as syntactically valid trees rather than randomized stacks.')
# Fitness Options
    parser.add_argument('-fit', action='store', dest='fit_type', default=None, choices = ['MSE','MAE','R2','VAF','combo','F1','F1W'],
                    type=str, help='Fitness metric (Default: mse). combo is mae/r2')

    parser.add_argument('--norm_error', action='store_true', dest='ERC_ints', default=None,
                help='Flag to normalize error by the standard deviation of the target data being used.')

    parser.add_argument('--fe', action='store_true', dest='estimate_fitness', default=None,
            help='Flag to coevolve fitness estimators.')

    parser.add_argument('-fe_p', action='store', dest='FE_pop_size', default=None, type = positive_integer,
                    help='fitness estimator population size.')

    parser.add_argument('-fe_i', action='store', dest='FE_ind_size', default=None, type = positive_integer,
                    help='number of fitness cases for FE to use.')

    parser.add_argument('-fe_t', action='store', dest='FE_train_size', default=None, type = positive_integer,
                    help='trainer population size. Trainers are evaluated on the entire data set.')

    parser.add_argument('-fe_g', action='store', dest='FE_train_gens', default=None, type = positive_integer,
                    help='Number of generations between trainer selections.')

    parser.add_argument('--fe_rank', action='store_true', dest='FE_rank', default=None,
            help='User rank for FE fitness rather than error.')

# Parameter hillclimbing
    parser.add_argument('--phc', action='store_true', dest='pHC_on', default=None,
                        help='Flag to use parameter hillclimbing.')

    parser.add_argument('-phc_its', action='store', dest='pHC_its', default=None, type = positive_integer,
                        help='Number of iterations of parameter hill climbing each generation.')
# Epigenetic hillclimbing
    parser.add_argument('--ehc', action='store_true', dest='eHC_on', default=None,
                        help='Flag to use epigenetic hillclimbing.')

    parser.add_argument('-ehc_its', action='store', dest='eHC_its', default=None, type = positive_integer,
                    help='Number of iterations of epigenetic hill climbing each generation.')

    parser.add_argument('-ehc_init', action='store', dest='eHC_init', default=None, type = positive_integer,
                        help='Fraction of initial population''s genes that are silenced.')

    parser.add_argument('--emut', action='store_true', dest='eHC_mut', default=None,
                        help='Flag to use epigenetic mutation. Only works if ehc flag is present.')

    parser.add_argument('--e_slim', action='store_true', dest='eHC_slim', default=None,
                        help='Flag to store partial program outputs such that point evaluations are minimized during eHC.')
# Pareto Archive
    parser.add_argument('--archive', action='store_true', dest='prto_arch_on', default=None,
                        help='Flag to save the Pareto front of equations (fitness and complexity) during the run.')

    parser.add_argument('-arch_size', action='store', dest='prto_arch_size', default=None,
                        help='Minimum size of the Pareto archive. By default, ellyn will save the entire front, but more individuals will be added if the front is less than arch_size.')
# island model
    parser.add_argument('--no_islands', action='store_false', dest='islands', default=True,
                    help='Flag to turn off island populations. Parallel execution across codes on a single node.')

    parser.add_argument('-num_islands', action='store', dest='num_islands', default=None, type = positive_integer,
                help='Number of islands to use (limits number of cores).')

    parser.add_argument('-island_gens', action='store', dest='island_gens', default=None, type = positive_integer,
                    help='Number of generations between synchronized shuffling of island populations.')
# lexicase selection options
    parser.add_argument('-lex_pool', action='store', dest='lex_pool', default=None,
                        type=float_range, help='Fraction of population to use in lexicase selection events [0.0, 1.0].')

    parser.add_argument('-lex_meta', action='store', dest='lex_meta', default=None, choices=['age','complexity'],
                    type=str,help='Specify extra cases for selection. Options: age, complexity.')

    parser.add_argument('--lex_eps_error_mad', action='store_true', dest='lex_eps_error_mad', default=None,
                    help='Flag to use epsilon lexicase with median absolute deviation, error-based epsilons.')

    parser.add_argument('--lex_eps_target_mad', action='store_true', dest='lex_eps_target_mad', default=None,
                    help='Flag to use epsilon lexicase with median absolute deviation, target-based epsilons.')

    parser.add_argument('--lex_eps_semidynamic', action='store_true', dest='lex_eps_semidynamic', default=None,
                    help='Flag to use dynamically defined error in epsilon lexicase selection.')

    parser.add_argument('--lex_eps_dynamic', action='store_true', dest='lex_eps_dynamic', default=None,
                    help='Flag to use dynamic error and epsilon in epsilon lexicase selection.')

    parser.add_argument('--lex_eps_dynamic_rand', action='store_true', dest='lex_eps_dynamic_rand', default=None,
                help='Flag to use dynamic error and random thresholds in epsilon lexicase selection.')

    parser.add_argument('--lex_eps_dynamic_madcap', action='store_true', dest='lex_eps_dynamic_madcap', default=None,
                help='Flag to use dynamic error and mad capped thresholds in epsilon lexicase selection.')

    parser.add_argument('--lex_age', action='store_true', dest='lexage', default=None,
                help='Flag to use age-fitness Pareto survival after lexicase selection.')
# SGD
    parser.add_argument('--SGD', action='store_true', dest='SGD', default=None,
                help='Use stochastic gradient descent to update program constant values.')

    parser.add_argument('-lr', action='store', dest='learning_rate', default=None,
                        type=float_range, help='Learning rate for stochastic gradient descent.')

    parser.add_argument('-s', action='store', dest='random_state', default=np.random.randint(4294967295),
                        type=int, help='Random number generator seed for reproducibility. Note that using multi-threading may '
                                       'make exacts results impossible to reproduce.')

    parser.add_argument('-v', action='store', dest='verbosity', default=0, choices=[0, 1, 2, 3],
                        type=int, help='How much information ellyn communicates while it is running: 0 = none, 1 = minimal, 2 = lots, 3 = all.')

    parser.add_argument('--no-update-check', action='store_true', dest='DISABLE_UPDATE_CHECK', default=False,
                        help='Flag indicating whether the ellyn version checker should be disabled.')

    # parser.add_argument('--version', action='version', version='ellyn {version}'.format(version=__version__),
    #                     help='Show ellyn\'s version number and exit.')

    args = parser.parse_args()
    #
    # if args.VERBOSITY >= 2:
    #     print('\nellyn settings:')
    #     for arg in sorted(args.__dict__):
    #         if arg == 'DISABLE_UPDATE_CHECK':
    #             continue
    #         print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
    #     print('')
    # pdb.set_trace()
    # load data from csv file
    if args.INPUT_SEPARATOR is None:
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR,engine='python')
    else: # use c engine for read_csv is separator is specified
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Label' in input_data.columns.values:
        input_data.rename(columns={'Label': 'label'}, inplace=True)

    input_data.rename(columns={'target': 'label', 'class': 'label'}, inplace=True)


    random_state = args.random_state if args.random_state > 0 else None

    if args.AR:
        train_i = input_data.index[:round(0.75*len(input_data.index))]
        test_i = input_data.index[round(0.75*len(input_data.index)):]
    else:
        train_i, test_i = train_test_split(input_data.index,
                                                        stratify = None,#  stratify=input_data['label'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=random_state)

    training_features = np.array(input_data.loc[train_i].drop('label', axis=1).values,dtype=np.float32, order='C')
    training_labels = np.array(input_data.loc[train_i, 'label'].values,dtype=np.float32,order='C')
    # training_features = np.array(input_data.drop('label', axis=1).values,dtype=np.float32, order='C')
    # training_labels = np.array(input_data['label'].values,dtype=np.float32,order='C')
    testing_features = input_data.loc[test_i].drop('label', axis=1).values
    testing_labels = input_data.loc[test_i, 'label'].values
    learner = ellyn(**args.__dict__)
    # learner = ellyn(generations=args.GENERATIONS, population_size=args.POPULATION_SIZE,
    #             mutation_rate=args.MUTATION_RATE, crossover_rate=args.CROSSOVER_RATE,
    #             machine_learner = args.MACHINE_LEARNER, min_depth = args.MIN_DEPTH,
    #             max_depth = args.MAX_DEPTH, sel = args.SEL, tourn_size = args.TOURN_SIZE,
    #             seed_with_ml = args.SEED_WITH_ML, op_weight = args.OP_WEIGHT,
    #             erc = args.ERC, random_state=args.random_state, verbosity=args.VERBOSITY,
    #             disable_update_check=args.DISABLE_UPDATE_CHECK,fit_choice = args.FIT_CHOICE)

    learner.fit(training_features, training_labels)

    if args.verbosity >= 1:
        print('\nTraining accuracy: {}'.format(learner.score(training_features, training_labels)))
        print('Holdout accuracy: {}'.format(learner.score(testing_features, testing_labels)))

    # if args.OUTPUT_FILE != '':
    #     learner.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
