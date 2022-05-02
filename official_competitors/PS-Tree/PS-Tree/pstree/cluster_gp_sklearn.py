import builtins
import operator
import sys
import time
import traceback
import warnings
from collections import deque, defaultdict
from itertools import compress

import numpy
import pyximport
from deap.gp import Terminal
from deap.tools import selNSGA2, selRandom, selSPEA2, selLexicase, selNSGA3
from icecream import ic
from scipy.stats import pearsonr, PearsonRConstantInputWarning, PearsonRNearConstantInputWarning
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sympy import parse_expr, Piecewise, srepr

from . import cluster_gp_tools
from .common_utils import gene_to_string, reset_random
from .custom_sklearn_tools import LassoRidge, RFERegressor
from .gp_visualization_utils import multigene_gp_to_string
from .multigene_gp import *

warnings.simplefilter("ignore", category=PearsonRConstantInputWarning)
warnings.simplefilter("ignore", category=PearsonRNearConstantInputWarning)
# warnings.simplefilter("ignore", category=RuntimeWarning)

pyximport.install(setup_args={"include_dirs": numpy.get_include()})

from deap import creator, base, tools, gp
from deap.algorithms import varAnd
from deap.base import Fitness
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree

from .cluster_gp_tools import add_pset_function, selAutomaticEpsilonLexicase, \
    selBestSum, selMOEAD, selIBEA, c_deepcopy
from .gp_function import *
from glmnet import ElasticNet


class FeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, compiled_individuals):
        self.compiled_individuals = compiled_individuals

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=None):
        all_features = []
        for func in self.compiled_individuals:
            yp = func(*X.T)
            if not isinstance(yp, np.ndarray) or yp.size == 1:
                yp = np.full(len(X), yp)
            all_features.append(np.squeeze(yp).reshape(-1, 1))
        all_features = np.concatenate(all_features, axis=1)
        return all_features


def train_normalization(func):
    def call(self, X, y, *param, **dict_param):
        if self.normalize:
            X = self.x_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(y.reshape(-1, 1)).squeeze()
        result = func(self, X, y, *param, **dict_param)
        return result

    return call


def get_labels(tree, X, soft_tree=False):
    if isinstance(tree, DecisionTreeClassifier) or isinstance(tree, KMeans) or \
        isinstance(tree, BayesianGaussianMixture) or isinstance(tree, GaussianNB) \
        or isinstance(tree, RandomForestClassifier) or isinstance(tree, LogisticRegression):
        if soft_tree:
            if hasattr(tree, 'predict_proba'):
                tree.labels_ = tree.predict_proba(X)
            else:
                tree.labels_ = tree.predict(X).astype(int)
        else:
            tree.labels_ = tree.predict(X).astype(int)
    elif isinstance(tree, DecisionTreeRegressor) or isinstance(tree, PseudoPartition):
        tree.labels_ = tree.apply(X)
    else:
        print(type(tree))
        raise Exception
    return tree.labels_


def predict_normalization(func):
    def call(self, X, y=None, *param, **dict_param):
        if self.normalize:
            X = self.x_scaler.transform(X)
        y_predict = func(self, X, y, *param, **dict_param)
        if self.normalize:
            y_predict = np.reshape(y_predict, (-1, 1))
            assert len(y_predict) == len(X)
            y_predict = self.y_scaler.inverse_transform(y_predict)
        return y_predict

    return call


class NormalizationRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, normalize=True, **params):
        self.normalize = normalize
        if normalize:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class FastMeasure(Fitness):
    def __init__(self, values=()):
        super().__init__(values)
        self._values = None

    def getValues(self):
        if self._values is None:
            self._values = tuple(map(operator.truediv, self.wvalues, self.weights))
        return self._values

    def setValues(self, values):
        try:
            self.wvalues = tuple(map(operator.mul, values, self.weights))
            self._values = tuple(map(operator.truediv, self.wvalues, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError("Both weights and assigned values must be a "
                            "sequence of numbers when assigning to values of "
                            "%r. Currently assigning value(s) %r of %r to a "
                            "fitness with weights %s."
                            % (self.__class__, values, type(values),
                               self.weights)).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()
        self._values = ()

    values = property(getValues, setValues, delValues,
                      ("Fitness values. Use directly ``individual.fitness.values = values`` "
                       "in order to set the fitness and ``del individual.fitness.values`` "
                       "in order to clear (invalidate) the fitness. The (unweighted) fitness "
                       "can be directly accessed via ``individual.fitness.values``."))


class EnsembleRidge(RidgeCV):
    def __init__(self, alphas=None):
        super().__init__()
        self.model = BaggingRegressor(RidgeCV(alphas=alphas), n_estimators=3)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y)
        self.coef_ = np.mean([m.coef_ for m in self.model.estimators_], axis=0)
        self.best_score_ = np.mean([m.best_score_ for m in self.model.estimators_], axis=0)
        return self

    def predict(self, X):
        return self.model.predict(X)


class GPRegressor(NormalizationRegressor):
    def __init__(self, input_names=None, n_pop=50, n_gen=200, max_arity=2, height_limit=6, constant_range=2,
                 cross_rate=0.9, mutate_rate=0.1, verbose=False, basic_primitive=True, gene_num=1, random_float=False,
                 log_dict_size=int(1e9), archive_size=None, category_num=1, cluster_gp=True,
                 select=selRandom, test_fun=None, train_test_fun=None, samples=20, min_samples_leaf=1,
                 max_depth=None, linear_scale=False, regression_type=None, regression_regularization=0,
                 score_function=None, validation_selection=True, ridge_alpha='np.logspace(0, 4)',
                 survival_selection='NSGA2', feature_normalization=True, structure_diversity=True,
                 space_partition_fun=None, adaptive_tree=True, original_features=True,
                 new_surrogate_function=True, advanced_elimination=True, super_object=None, final_prune='Lasso',
                 correlation_elimination=False, tree_shrinkage=False, size_objective=True, soft_label=False,
                 initial_height=None, **params):
        """
        :param n_pop: size of population
        :param n_gen: number of generations
        """
        super().__init__(**params)
        self.initial_height = initial_height
        self.soft_label = soft_label
        self.average_size = sys.maxsize
        self.advanced_elimination = advanced_elimination
        self.new_surrogate_function = new_surrogate_function
        self.structure_diversity = structure_diversity
        self.adaptive_tree = adaptive_tree
        self.validation_selection = validation_selection
        self.ridge_alpha = ridge_alpha
        self.score_function = score_function
        self.regression_type = regression_type
        self.regression_regularization = regression_regularization
        if hasattr(creator, 'FitnessMin'):
            del creator.FitnessMin
        if hasattr(creator, 'Individual'):
            del creator.Individual

        self.toolbox = None
        self.category_num = category_num
        # "cluster_gp" is the decisive control parameter
        self.cluster_gp = cluster_gp
        self.test_fun = test_fun
        self.train_test_fun = train_test_fun
        self.samples = samples
        self.best_distribution_test = None
        self.linear_scale = linear_scale
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.accuracy_list = []
        self.select = select
        self.pipelines = []
        self.best_cv = np.inf
        self.best_pop = None
        self.best_leaf_node_num = None
        self.best_tree = None
        self.survival_selection = survival_selection
        self.archive_size = archive_size
        self.input_names = input_names
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.max_arity = max_arity
        self.verbose = verbose
        self.basic_primitive = basic_primitive
        self.params = params
        self.height_limit = height_limit
        self.constant_range = constant_range
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.gene_num = gene_num
        self.random_float = random_float
        self.log_dict_size = log_dict_size
        self.feature_normalization = feature_normalization
        self.space_partition_fun = space_partition_fun
        self.original_features = original_features
        self.update_iteration = []
        self.current_gen = 0
        self.better_pop_flag = False
        self.super_object: PSTreeRegressor = super_object
        self.last_loss = None
        self.final_prune = final_prune
        self.size_objective = size_objective
        self.tree_shrinkage = tree_shrinkage
        self.correlation_elimination = correlation_elimination

    def get_predicted_list(self, pop):
        predicted_list = []
        for ind in pop:
            predicted_list.append(ind.predicted_value)
        return predicted_list

    def evaluate(self, individuals, final_model=None):
        compiled_individuals = [self.toolbox.compile(individual) for individual in individuals]
        all_features = self.feature_construction(compiled_individuals, self.train_data)
        fitness, pipelines, score = self.model_construction(all_features, final_model)

        if self.verbose:
            print('score', score / len(self.Y))

        # correlation = np.corrcoef(np.array([p['Ridge'].coef_ for p in pipelines]))
        self.adaptive_tree_generation(self.feature_construction(compiled_individuals, self.train_data,
                                                                self.original_features),
                                      pipelines)

        if (self.validation_selection and score < self.best_cv) or (not self.validation_selection) or \
            (final_model != None):
            # record the best individual in the training process
            self.update_iteration.append((self.current_gen, score / len(self.Y)))
            self.best_cv = score
            self.pipelines = pipelines
            self.best_pop = individuals[:]
            self.better_pop_flag = True
            if self.adaptive_tree:
                self.best_features = all_features
                self.best_label = self.category
                self.best_leaf_node_num = self.super_object.max_leaf_nodes
            # assert len(pipelines) == category_num + 1, f"{category_num + 1},{len(pipelines)}"

        fitness = np.array(fitness)
        assert len(fitness.shape) == 2, fitness.shape
        fitness_dimension = len(pipelines)

        for i, ind in enumerate(individuals):
            if self.size_objective:
                target_dimension = fitness_dimension + 1
                ind.fitness.weights = tuple([1 for _ in range(target_dimension)])
                ind.fitness.values = tuple(np.abs(fitness[:, i])) + \
                                     (-0.01 * max(len(ind), self.average_size) / self.average_size,)
            else:
                target_dimension = fitness_dimension
                ind.fitness.weights = tuple([1 for _ in range(target_dimension)])
                ind.fitness.values = tuple(np.abs(fitness[:, i]))
            assert len(ind.fitness.values) == target_dimension
            assert len(ind.fitness.wvalues) == target_dimension
        return tuple(fitness)

    def model_construction(self, all_features, final_model):
        fitness = []
        pipelines = []
        score = 0
        if len(self.category.shape) == 1:
            category_num = np.max(self.category)
        else:
            category_num = self.category.shape[1] - 1

        def dummy_regressor_construction(x, y):
            coef = np.zeros(x.shape[1])
            constant = np.mean(y)
            regr = Pipeline(
                [
                    ("Scaler", StandardScaler()),
                    ("Ridge", DummyRegressor(strategy='constant', constant=constant)),
                ]
            )
            regr.fit(features, y)
            regr['Ridge'].coef_ = coef
            regr['Ridge'].intercept_ = constant
            # append coefficients and pipelines to the archive
            fitness.append(coef)
            pipelines.append(regr)
            return coef, regr

        for i in range(category_num + 1):
            def check_rule(x):
                # if number of samples <2 :unable to execute leave-one CV
                # if number of samples <10 :unable to execute 5-fold CV
                if x < 2 or (x < 10 and not (self.ridge_alpha == 'RidgeCV' and final_model == None)):
                    return True
                else:
                    return False

            if len(self.category.shape) == 1:
                category = self.category == i
                Y_true = self.Y[category]
                features = all_features[category]

                if check_rule(np.sum(category)):
                    dummy_regressor_construction(features, Y_true)
                    continue
            else:
                # soft decision tree
                Y_true = self.Y
                features = all_features

                if check_rule(np.count_nonzero(self.category[:, i])):
                    dummy_regressor_construction(features, Y_true)
                    continue

            # if (np.sum(category) < all_features.shape[1] and self.adaptive_tree) or (np.sum(category) < 5):
            # warnings.simplefilter("ignore", category=ConvergenceWarning)

            def get_lasso():
                alphas = _alpha_grid(features, Y_true, normalize=True)
                ridge_model = ElasticNet(alpha=1, lambda_path=alphas, n_splits=5, tol=1e-4,
                                         random_state=0)
                return ridge_model

            def get_elastic_net(ratio):
                alphas = _alpha_grid(features, Y_true, l1_ratio=ratio, normalize=True)
                ridge_model = ElasticNet(alpha=ratio, lambda_path=alphas, n_splits=5, tol=1e-4,
                                         random_state=0)
                return ridge_model

            # determine the evaluation model
            if self.ridge_alpha == 'Lasso':
                ridge_model = get_lasso()
            elif self.ridge_alpha == 'Linear':
                ridge_model = LinearRegression()
            elif 'ElasticNet' in self.ridge_alpha:
                ratio = float(self.ridge_alpha.split('-')[1])
                ridge_model = get_elastic_net(ratio)
            elif self.ridge_alpha == 'LinearSVR':
                ridge_model = LinearSVR()
            elif self.ridge_alpha == 'EnsembleRidge':
                ridge_model = EnsembleRidge(np.logspace(0, 4))
            else:
                # default use this option
                ridge_model = RidgeCV(alphas=eval(self.ridge_alpha))

            # determine the final model
            if final_model == 'Lasso':
                # default use this option
                ridge_model = get_lasso()
            elif final_model == 'ElasticNet':
                ridge_model = get_elastic_net(0.5)
            elif final_model == 'LassoRidge':
                ridge_model = LassoRidge(get_lasso(), ridge_model)
            elif final_model == 'RFE':
                ridge_model = RFERegressor(get_lasso(), n_features_to_select=10, step=5)

            if self.feature_normalization:
                steps = [
                    ("Scaler", StandardScaler()),
                    ("Ridge", ridge_model),
                ]
                pipe = Pipeline(steps)
            else:
                pipe = Pipeline([
                    ("Ridge", ridge_model),
                ])

            if self.validation_selection:
                # record the best individual in the training process
                ridge: RidgeCV = pipe["Ridge"]
                try:
                    if len(self.category.shape) == 1:
                        pipe.fit(features, Y_true)
                    else:
                        weight = np.nan_to_num(self.category[:, i], posinf=0, neginf=0)
                        pipe.fit(features, Y_true, Ridge__sample_weight=weight)
                except Exception as e:
                    traceback.print_exc()
                    ic(e, features.shape, Y_true.shape)
                    # not converge
                    dummy_regressor_construction(features, Y_true)
                    continue
                if isinstance(ridge, RidgeCV):
                    if len(self.category.shape) == 1:
                        score += abs(len(Y_true) * ridge.best_score_)
                    else:
                        score += abs(np.sum(self.category[:, i]) * ridge.best_score_)
                elif isinstance(ridge, ElasticNet):
                    if len(self.category.shape) == 1:
                        score += -1 * abs(len(Y_true) * np.max(ridge.cv_mean_score_))
                    else:
                        score += -1 * abs(np.sum(self.category[:, i]) * np.max(ridge.cv_mean_score_))
                elif isinstance(ridge, LassoCV):
                    score += abs(len(Y_true) * np.min(np.sum(ridge.mse_path_, axis=1)))
                elif isinstance(ridge, ElasticNetCV):
                    score += abs(len(Y_true) * np.min(np.sum(ridge.mse_path_, axis=1)))
                elif isinstance(ridge, RFERegressor):
                    score += 0
                elif isinstance(ridge, LassoRidge):
                    score += 0
                else:
                    raise Exception
                if isinstance(ridge, ElasticNet):
                    feature_importances = np.mean(np.abs(ridge.coef_path_), axis=1)
                else:
                    feature_importances = np.abs(ridge.coef_)
            else:
                pipe.fit(features, np.squeeze(Y_true))
                feature_importances = np.abs(pipe['Ridge'].coef_)
            fitness.append(feature_importances)
            pipelines.append(pipe)
        return fitness, pipelines, score

    def feature_construction(self, compiled_individuals, x, original_features=False):
        # construct all features
        if original_features:
            all_features = [x]
        else:
            all_features = []
        for func in compiled_individuals:
            yp = func(*x.T)
            if not isinstance(yp, np.ndarray) or yp.size == 1:
                yp = np.full(len(x), yp)
            all_features.append(np.squeeze(yp).reshape(-1, 1))
        all_features = np.concatenate(all_features, axis=1)
        all_features = np.nan_to_num(all_features, posinf=0, neginf=0)
        return all_features

    def adaptive_tree_generation(self, all_features, pipelines):
        if self.adaptive_tree:
            if self.soft_label:
                original_all_features = all_features
                prob = softmax(np.array([(p.predict(all_features[:, self.train_data.shape[1]:])
                                          - self.Y) ** 2 * -1 for p in pipelines]),
                               axis=0)
                sample = np.random.rand(len(pipelines), all_features.shape[0])
                matrix = prob > sample
                features = np.concatenate([all_features[s] for s in matrix], axis=0)
                label = np.concatenate([np.full(np.sum(s == True), i) for i, s in enumerate(matrix)], axis=0)
                all_features = features
                _, decision_tree = self.space_partition_fun(all_features, label)
                self.category = decision_tree.predict_proba(original_all_features)
                # decision_tree.labels_ = self.category
            else:
                # assign data point to new partitions
                label = np.zeros(len(self.Y))
                best_fitness = np.full(len(self.Y), np.inf)
                for i, p in enumerate(pipelines):
                    # np.array([(p.predict(all_features[:, self.train_data.shape[1]:]) - self.Y) ** 2 for p in pipelines])
                    if self.original_features:
                        loss = (p.predict(all_features[:, self.train_data.shape[1]:]) - self.Y) ** 2
                    else:
                        loss = (p.predict(all_features) - self.Y) ** 2
                    label[loss < best_fitness] = i
                    best_fitness[loss < best_fitness] = loss[loss < best_fitness]
                self.category, decision_tree = self.space_partition_fun(all_features, label)

    def statistic_fun(self, ind):
        # return loss and time
        if self.test_fun is not None:
            if not self.better_pop_flag:
                return self.last_loss
            self.better_pop_flag = False
            train_test_loss = self.train_test_fun.predict_loss()
            test_loss = self.test_fun.predict_loss()
            self.last_loss = (train_test_loss, time.time(), test_loss)
            return self.last_loss
        return (time.time(),)

    def fit(self, X, y=None, category=None):
        if not hasattr(self, 'fit_function'):
            raise Exception("Fit function must be specified!")

        if (not hasattr(self, 'input_names')) or (self.input_names is None):
            self.input_names = [f'X{i}' for i in range(X.shape[1])]

        self.train_data = X
        self.Y = y

        verbose = self.verbose
        if verbose:
            self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
            self.stats.register("avg", np.mean, axis=0)
            self.stats.register("std", np.std, axis=0)
            self.stats.register("min", np.min, axis=0)
            self.stats.register("max", np.max, axis=0)
        else:
            self.stats = tools.Statistics(key=self.statistic_fun)
            self.stats.register("min", np.min, axis=0)
            self.stats.register("max", np.max, axis=0)

        backup_X = X.copy()
        backup_y = y.copy()
        self.lazy_init(self.input_names)

        if category is None:
            category = np.full([y.shape[0]], 0)

        self.category = category

        self.fit_function()

        assert np.all(backup_X == X), "Data has been changed unexpected!"
        assert np.all(backup_y == y), "Data has been changed unexpected!"
        return self

    def feature_synthesis(self, x, pop, original_features=False):
        compiled_pop = [self.toolbox.compile(individual) for individual in pop]
        return self.feature_construction(compiled_pop, x, original_features)

    def predict(self, X, y=None, category=None):
        # save_object([str(x) for x in self.hof.items], 'model.pkl')
        if (category is None) or (not self.cluster_gp):
            category = np.full([X.shape[0]], 0)

        Yp = np.zeros(X.shape[0])
        if len(category.shape) == 1:
            category_num = np.max(category)
        else:
            category_num = category.shape[1] - 1

        X = self.feature_synthesis(X, self.best_pop)
        for i in range(category_num + 1):
            if len(category.shape) == 1:
                loc = np.where(category == i)
                current_c = category == i
                if np.sum(current_c) == -0:
                    continue
                features = X[current_c]
            else:
                features = X

            if len(features.shape) == 1:
                features = features.reshape(1, len(features))
            assert features.shape[1] >= len(self.best_pop), features.shape[1]
            if len(category.shape) == 1:
                Yp.put(loc, self.pipelines[i].predict(features))
            else:
                Yp += np.multiply(self.pipelines[i].predict(features), category[:, i])
        return Yp

    def __deepcopy__(self, memodict={}):
        return c_deepcopy(self)

    def lazy_init(self, input_names):
        pset = gp.PrimitiveSet("MAIN", len(input_names), prefix='X')
        toolbox = base.Toolbox()
        toolbox.register('evaluate', self.evaluate)
        toolbox.register('select', self.select)

        self.pset = pset
        self.toolbox = toolbox

        add_pset_function(pset, self.max_arity, self.basic_primitive)
        if hasattr(gp, 'rand101'):
            # delete existing constant generator
            delattr(gp, 'rand101')
        if self.random_float:
            pset.addEphemeralConstant('rand101', lambda: random.uniform(-self.constant_range, self.constant_range))
        else:
            pset.addEphemeralConstant("rand101", lambda: random.randint(-self.constant_range, self.constant_range))

        creator.create("FitnessMin", FastMeasure, weights=tuple([1 for _ in range(self.category_num)]))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        if self.initial_height is None:
            toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=2)
        else:
            a, b = self.initial_height.split('-')
            a, b = int(a), int(b)
            toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=a, max_=b)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.height_limit))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.height_limit))
        if self.n_gen == 0:
            self.pop = self.generation_original_features()
        else:
            self.pop = toolbox.population(n=self.n_pop)

    def generation_original_features(self):
        pop = []
        for x in self.pset.terminals[object]:
            if type(x) is Terminal:
                tree = gp.PrimitiveTree([x])
                tree.fitness = creator.FitnessMin()
                pop.append(tree)
        assert len(pop) == self.train_data.shape[1]
        return pop

    def fit_function(self):
        self.pop, self.log_book = self.moea(self.pop, self.toolbox,
                                            self.cross_rate, self.mutate_rate,
                                            self.n_gen, stats=self.stats,
                                            halloffame=None, verbose=self.verbose,
                                            params=self.params)

    def moea(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, params=None):
        if self.new_surrogate_function is True:
            def individual_to_tuple(ind):
                return tuple(self.feature_synthesis(self.train_data[:20], [ind]).flatten().tolist())
        elif str(self.new_surrogate_function).startswith('First'):
            sample_count = int(self.new_surrogate_function.split('-')[1])

            def individual_to_tuple(ind):
                return tuple(self.feature_synthesis(self.train_data[:sample_count], [ind]).flatten().tolist())
        else:
            individual_to_tuple = cluster_gp_tools.individual_to_tuple

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        toolbox.evaluate(population)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        diversity_list = []
        pop_size_list = []
        log_dict = LogDict(self.log_dict_size * len(population))
        for p in population:
            ind_tuple = individual_to_tuple(p)
            p.ind_tuple = ind_tuple
            log_dict.insert(ind_tuple, p.fitness.values)
        pop_size = len(population)
        # assigning the crowding distance to each individual
        if self.select == selTournamentDCD and self.survival_selection == 'NSGA2':
            population = selNSGA2(population, pop_size)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            if self.basic_primitive == 'dynamic' and self.current_gen > (self.n_gen // 2):
                self.pset.addPrimitive(np.sin, 1)
                self.pset.addPrimitive(np.cos, 1)
            if self.tree_shrinkage and (gen % 50) == 0:
                self.super_object.max_leaf_nodes = max(self.super_object.max_leaf_nodes // 2, 1)
            self.current_gen = gen
            if self.structure_diversity:
                count = 0
                new_offspring = []
                while (len(new_offspring) < pop_size):
                    count += 1
                    # Select the next generation individuals
                    # if self.survival_selection == 'Random':
                    #     offspring = selRandom(population, 2)
                    # else:
                    offspring = toolbox.select(population, 2)
                    offspring = offspring[:]
                    parent_tuple = [o.ind_tuple for o in offspring]
                    # Vary the pool of individuals
                    offspring = varAnd(offspring, toolbox, cxpb, mutpb)
                    for o in offspring:
                        if len(new_offspring) < pop_size:
                            ind_tuple = individual_to_tuple(o)
                            if count > pop_size * 50:
                                # if too many trials failed, then we just allow to use repetitive individuals
                                o.ind_tuple = ind_tuple
                                log_dict.insert(ind_tuple, -1)
                                new_offspring.append(o)
                                continue
                            if not log_dict.exist(ind_tuple):
                                if self.advanced_elimination and (
                                    np.abs(pearsonr(ind_tuple, parent_tuple[0])[0]) >= 0.95 or
                                    np.abs(pearsonr(ind_tuple, parent_tuple[1])[0]) >= 0.95):
                                    log_dict.insert(ind_tuple, -1)
                                    continue
                                o.ind_tuple = ind_tuple
                                log_dict.insert(ind_tuple, -1)
                                new_offspring.append(o)

                offspring = new_offspring

            else:
                offspring = toolbox.select(population, len(population))
                # Vary the pool of individuals
                offspring = varAnd(offspring, toolbox, cxpb, mutpb)
            assert len(offspring) == pop_size, print(len(offspring), pop_size)
            self.average_size = np.mean([len(p) for p in population])

            # Evaluate the individuals with an invalid fitness
            if self.correlation_elimination:
                corr_matrix = np.abs(np.corrcoef(np.array([p.ind_tuple for p in population + offspring])))
                # Select upper triangle of correlation matrix
                upper = np.triu(corr_matrix, k=1)
                # Find index of feature columns with correlation greater than 0.95
                to_drop = [any(upper[i] > 0.95) for i in range(0, upper.shape[0])]
                parent = list(compress(population + offspring, np.invert(to_drop)))
                toolbox.evaluate(parent)
            else:
                toolbox.evaluate(offspring + population)

            # diversity = diversity_measure(offspring) / pop_size
            # diversity_list.append(diversity)
            pop_size_list.append(len(offspring))
            log_dict.gc()

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            if self.survival_selection == 'NSGA2':
                # if len(offspring[0].fitness.wvalues) > 2:
                #     high_dimensional = True
                #     self.random_objectives = np.random.uniform(0, 1, size=(len(offspring[0].fitness.wvalues), 2))
                #     for ind in offspring + population:
                #         setattr(ind, 'original_fitness', ind.fitness.values)
                #         setattr(ind, 'original_weights', ind.fitness.weights)
                #         fitness = np.array(ind.fitness.wvalues) @ self.random_objectives
                #         ind.fitness.weights = (1,) * len(fitness)
                #         ind.fitness.values = list(fitness)
                # else:
                #     high_dimensional = False
                if self.correlation_elimination:
                    population[:] = selNSGA2(parent, pop_size)
                else:
                    population[:] = selNSGA2(population + offspring, pop_size)
                    # population = list(filter(lambda x: np.sum(x.fitness.wvalues) > 0, population))
                # if high_dimensional:
                #     for ind in population:
                #         ind.fitness.weights = getattr(ind, 'original_weights')
                #         ind.fitness.values = getattr(ind, 'original_fitness')
            elif self.survival_selection == 'IBEA':
                population[:] = selIBEA(population + offspring, pop_size)
            elif self.survival_selection == 'SPEA2':
                population[:] = selSPEA2(population + offspring, pop_size)
            elif self.survival_selection == 'NSGA3':
                ref_points = tools.uniform_reference_points(nobj=len(population[0].fitness.wvalues))
                population[:] = selNSGA3(population + offspring, pop_size, ref_points)
            elif self.survival_selection == 'Lexicase':
                def selLexicasePlus(individuals: list, k: int):
                    selected_individuals = []
                    while len(selected_individuals) < k:
                        lexicase_inds = selLexicase(individuals, 1)
                        for x in lexicase_inds:
                            individuals.remove(x)
                        selected_individuals.extend(lexicase_inds)
                    return selected_individuals

                population[:] = selLexicasePlus(population + offspring, pop_size)
            elif self.survival_selection == 'AutomaticEpsilonLexicase':
                def selAutomaticEpsilonLexicasePlus(individuals: list, k: int):
                    selected_individuals = []
                    while len(selected_individuals) < k:
                        lexicase_inds = selAutomaticEpsilonLexicase(individuals, 1)
                        for x in lexicase_inds:
                            individuals.remove(x)
                        selected_individuals.extend(lexicase_inds)
                    return selected_individuals

                population[:] = selAutomaticEpsilonLexicasePlus(population + offspring, pop_size)
            elif self.survival_selection == 'Random':
                def selSample(individuals, k):
                    return random.sample(individuals, k)

                population[:] = selSample(population + offspring, pop_size)
            elif self.survival_selection == 'Best':
                population[:] = selBestSum(population + offspring, pop_size)
            elif self.survival_selection == 'MOEA/D':
                population[:] = selMOEAD(population + offspring, pop_size)
            else:
                raise Exception
            assert len(population) <= pop_size

            # if self.test_fun != None:
            #     # reevaluate population
            #     self.evaluate(population)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

        # final process
        # select top-N individuals
        # selNSGA2(population, len(population))
        # assert len(population) == pop_size
        # toolbox.evaluate(self.best_pop, final_model=True)

        self.super_object.max_leaf_nodes = self.best_leaf_node_num
        self.super_object.soft_tree = self.super_object.final_soft_tree
        if self.final_prune is not None:
            toolbox.evaluate(self.best_pop, final_model=self.final_prune)

        features = self.feature_synthesis(self.train_data, self.best_pop,
                                          self.original_features)
        self.adaptive_tree_generation(features, self.pipelines)
        return population, logbook


class NormalizedGPRegressor(GPRegressor):
    def __init__(self, **params):
        super().__init__(**params)

    @train_normalization
    def fit(self, X, y=None, category=None):
        super().fit(X, y, category)

    @predict_normalization
    def predict(self, X, y=None, category=None):
        super().predict(X, y, category)


def diversity_measure(pop):
    fit = set()
    for ind in pop:
        fit.add(ind.fitness.values[0])
    return len(fit)


class PSTreeRegressor(NormalizationRegressor):
    """
    An upper-level class for PS-Tree
    """

    def __init__(self, regr_class, tree_class, min_samples_leaf=1, max_depth=None, max_leaf_nodes=4, random_seed=0,
                 restricted_classification_tree=True, basic_primitive='optimal',
                 soft_tree=True, final_soft_tree=True, adaptive_tree=True, random_state=0, **params):
        """
        regr_class: the class name for base learner
        tree_class: the class name for the upper-level decision tree
        """
        super().__init__(**params)
        self.random_state = random_state
        reset_random(self.random_state)
        self.regr_class = regr_class
        self.tree_class = tree_class
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.random_seed = random_seed
        self.params = params
        self.restricted_classification_tree = restricted_classification_tree
        self.basic_primitive = basic_primitive
        self.soft_tree = soft_tree
        self.final_soft_tree = soft_tree & final_soft_tree
        self.adaptive_tree = adaptive_tree

    @train_normalization
    def fit(self, X: np.ndarray, y=None):
        self.train_data = X
        self.train_label = y
        if self.min_samples_leaf in ['Auto', 'Auto-4', 'Auto-6', 'Auto-8']:
            best_size = automatically_determine_best_size(X, y, self.min_samples_leaf)
            self.min_samples_leaf = best_size
        if type(self.min_samples_leaf) is str:
            raise Exception

        category, _ = self.space_partition(X, y)
        if self.adaptive_tree is True:
            self.tree_class = DecisionTreeClassifier
        if self.adaptive_tree == 'Soft':
            self.tree_class = LogisticRegression

        self.regr: GPRegressor = self.regr_class(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                                 space_partition_fun=self.space_partition,
                                                 basic_primitive=self.basic_primitive, soft_tree=self.soft_tree,
                                                 adaptive_tree=self.adaptive_tree, super_object=self, **self.params)
        self.regr.fit(X, y, category)
        return self

    def space_partition(self, X, y):
        # other partition methods
        if self.tree_class == PseudoPartition or self.max_leaf_nodes == 1:
            self.tree = PseudoPartition()
            # return self.tree.apply(X), self.tree
        elif self.tree_class == DecisionTreeClassifier:
            if self.restricted_classification_tree:
                self.tree = self.tree_class(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=self.random_seed)
            else:
                self.tree = self.tree_class(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_seed,
                    ccp_alpha=0.01)
        elif self.tree_class == LogisticRegression:
            self.tree = LogisticRegression(
                solver='liblinear'
            )
        elif self.tree_class == DecisionTreeRegressor:
            self.tree = self.tree_class(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=self.random_seed)
        elif self.tree_class == KMeans:
            self.tree = KMeans(n_clusters=self.max_leaf_nodes, random_state=self.random_seed)
        elif self.tree_class == BayesianGaussianMixture:
            self.tree = BayesianGaussianMixture(n_components=self.max_leaf_nodes, max_iter=1000,
                                                random_state=self.random_seed)
        elif self.tree_class == GaussianNB:
            self.tree = GaussianNB()
        elif self.tree_class == RandomForestClassifier:
            self.tree = RandomForestClassifier(n_estimators=10,
                                               max_depth=self.max_depth,
                                               min_samples_leaf=self.min_samples_leaf,
                                               max_leaf_nodes=self.max_leaf_nodes,
                                               random_state=self.random_seed)
        else:
            raise Exception

        if hasattr(self, 'regr') and self.regr.original_features == 'original':
            self.tree.fit(X[:, :self.train_data.shape[1]], y)
            self.tree.labels_ = get_labels(self.tree, X[:, :self.train_data.shape[1]],
                                           self.soft_tree)
        else:
            self.tree.fit(X, y)
            self.tree.labels_ = get_labels(self.tree, X, self.soft_tree)
        if len(self.tree.labels_.shape) == 1:
            cluster_num = self.tree.labels_.max() + 1
            category, category_index = self.category_generation(cluster_num, y)
            self.params['category_num'] = category_index
        else:
            category = self.tree.labels_
            self.params['category_num'] = category.shape[1]
        # if isinstance(self.tree, DecisionTreeClassifier):
        #     print('loss', accuracy_score(y, self.tree.predict(X)), np.unique(self.tree.labels_).__len__())
        # if isinstance(self.tree, DecisionTreeRegressor):
        #     print('loss', r2_score(y, self.tree.predict(X)), np.unique(self.tree.labels_).__len__())
        return category, self.tree

    def category_generation(self, cluster_num, y):
        category = np.full([y.shape[0]], 0)
        category_index = 0
        self.label_map = {}
        for i in range(cluster_num):
            if not np.any(self.tree.labels_ == i):
                continue
            category[np.where(self.tree.labels_ == i)] = category_index
            self.label_map[i] = category_index
            category_index += 1
        return category, category_index

    @predict_normalization
    def predict(self, X, y=None):
        if self.regr.adaptive_tree:
            features = self.regr.feature_synthesis(X, self.regr.best_pop,
                                                   original_features=self.regr.original_features)
            if self.regr.original_features == 'original':
                labels = get_labels(self.tree, features[:, :self.train_data.shape[1]])
            else:
                labels = get_labels(self.tree, features, self.soft_tree)
        else:
            labels = get_labels(self.tree, X)
        backup_X = X.copy()

        if len(labels.shape) == 1:
            labels = self.category_map(labels)

        y_predict = self.regr.predict(X, y, category=labels)

        assert np.all(backup_X == X), "Data has been changed unexpected!"
        return y_predict

    def category_map(self, labels):
        for i, label in enumerate(labels):
            if label in self.label_map.keys():
                labels[i] = self.label_map[label]
            else:
                # The untrained cluster.
                labels[i] = -1
        return labels

    def __deepcopy__(self, memodict={}):
        return c_deepcopy(self)

    def get_params(self, deep=True):
        # The current version of scikit-learn does not support this function well,
        # which cause an error when the constructor parameter has a parameter that is an estimator class.
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params') and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value

        params = out
        params = {
            **self.params,
            **params
        }
        return params

    def model(self, partition=0):
        features = []
        for id in range(self.regr.train_data.shape[1]):
            features.append(parse_expr(f'X{id}'))
        for p in self.regr.best_pop:
            features.append(parse_expr(gene_to_string(p)))

        regr, feature_names = self, [f'X{id}' for id in range(self.regr.train_data.shape[1] + len(self.regr.best_pop))]
        tree_ = regr.tree.tree_
        feature_name = [feature_names[i]
                        if i != _tree.TREE_UNDEFINED else "undefined!"
                        for i in tree_.feature]

        if regr.tree.tree_.node_count == 1:
            # single model
            return srepr(multigene_gp_to_string(0, regr.regr))

        all_expressions = []
        all_conditions = []

        def recurse(node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                index = int(name.replace('X', ''))
                name = features[index]
                threshold = tree_.threshold[node]

                node_condition = f'({name} <= {threshold})'
                all_conditions.append(node_condition)
                recurse(tree_.children_left[node])
                all_conditions.pop(-1)

                node_condition = f'({name} > {threshold})'
                all_conditions.append(node_condition)
                recurse(tree_.children_right[node])
                all_conditions.pop(-1)
            else:
                tree_values = tree_.value[node][0]
                tree_values = tree_values / tree_values.sum()
                # print(node)
                expr = None
                for i in range(len(regr.regr.pipelines)):
                    ex1 = multigene_gp_to_string(i, regr.regr)
                    if expr is None:
                        expr = tree_values[i] * ex1
                    else:
                        expr += tree_values[i] * ex1
                condition = '&'.join(all_conditions)
                all_expressions.append((expr, parse_expr(condition)))

        recurse(0)
        return srepr(Piecewise(*tuple(all_expressions)))


class SequentialTreeGPRegressor(NormalizationRegressor):
    def __init__(self, regr_class, min_samples_leaf=1, min_impurity_decrease=0, random_seed=0, **params):
        super().__init__(**params)
        self.regr_class = regr_class
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_seed = random_seed
        self.params = params

    @train_normalization
    def fit(self, X: np.ndarray, y=None):
        self.tree = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                          min_impurity_decrease=self.min_impurity_decrease,
                                          random_state=self.random_seed)
        self.tree.fit(X, y)
        self.tree.labels_ = self.tree.apply(X)
        cluster_num = self.tree.labels_.max() + 1

        category_index = 0
        self.label_map = {}
        self.regr = []

        if 'test_fun' in self.params and self.params['test_fun'] is not None:
            test_fun = self.params['test_fun']
            self.test_x = test_fun.x
            self.test_y = test_fun.y
            test_label = self.tree.apply(self.test_x)

            train_test_fun = self.params['train_test_fun']
            self.train_x = train_test_fun.x
            self.train_y = train_test_fun.y
            train_label = self.tree.apply(self.train_x)

        for i in range(cluster_num):
            if not np.any(self.tree.labels_ == i):
                continue

            if 'test_fun' in self.params and self.params['test_fun'] is not None:
                test_fun = self.params['test_fun']
                test_fun.x = self.test_x[test_label == i]
                test_fun.y = self.test_y[test_label == i]
                self.params['test_fun'] = test_fun

                train_test_fun = self.params['train_test_fun']
                train_test_fun.x = self.train_x[train_label == i]
                train_test_fun.y = self.train_y[train_label == i]
                self.params['train_test_fun'] = train_test_fun
            self.label_map[i] = category_index
            category_index += 1

            regr = self.regr_class(category_num=1, **self.params)
            self.regr.append(regr)
            regr.fit(X[self.tree.labels_ == i], y[self.tree.labels_ == i],
                     np.zeros((np.sum(self.tree.labels_ == i, )), dtype=int))
        return self

    @predict_normalization
    def predict(self, X, y=None):
        labels = self.tree.apply(X)
        for i, label in enumerate(labels):
            if label in self.label_map.keys():
                labels[i] = self.label_map[label]
            else:
                # The untrained cluster.
                labels[i] = -1

        y_predict = np.zeros((X.shape[0],))
        for i in range(len(self.regr)):
            if np.sum(labels == i) > 0:
                y_predict[labels == i] = self.regr[i].predict(X[labels == i],
                                                              category=np.zeros((np.sum(labels == i),), dtype=int))
        return y_predict

    def __deepcopy__(self, memodict={}):
        return c_deepcopy(self)

    def get_params(self, deep=True):
        # The current version of scikit-learn does not support this function well,
        # which cause an error when the constructor parameter has a parameter that is an estimator class.
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params') and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value

        params = out
        params = {
            **self.params,
            **params
        }
        return params


class PseudoCluster(BaseEstimator):
    def __init__(self, n_clusters=10, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        pass

    def predict(self, X, y=None):
        total_size = X.shape[0]
        cluster_size = total_size // self.n_clusters
        remain_size = total_size % self.n_clusters
        labels = np.array([], dtype=np.int)
        for i in range(self.n_clusters):
            labels = np.append(labels, np.full(cluster_size, i, dtype=np.int))
            if i < remain_size:
                labels = np.append(labels, np.full(1, i, dtype=np.int))
        np.random.shuffle(labels)
        return labels

    def __deepcopy__(self, memodict={}):
        return c_deepcopy(self)


class LogDict:
    def __init__(self, max_size):
        self.fitness_dict = {}
        self.queue = deque()
        self.max_size = max_size
        return

    def insert(self, x, fitness):
        if x not in self.fitness_dict:
            self.queue.append(x)
            self.fitness_dict[x] = fitness

    def gc(self):
        while len(self.queue) >= self.max_size:
            old = self.queue.popleft()
            del self.fitness_dict[old]

    def update(self, x, fitness):
        self.fitness_dict[x] = fitness

    def exist(self, x):
        return x in self.fitness_dict.keys()

    def get(self, x):
        if x in self.fitness_dict.keys():
            return self.fitness_dict[x]
        else:
            return None


class PseudoPartition(BaseEstimator):
    def __init__(self, **param):
        class zero: pass

        self.tree_ = zero()
        # this is because node count is the number of all nodes
        # no branch nodes exist in this tree
        setattr(self.tree_, 'node_count', 1)
        # there is only one leaf node in this pseudo tree
        setattr(self.tree_, 'n_leaves', 1)

    def fit(self, X, y=None):
        return np.zeros(len(X)).astype(np.int)

    def predict(self, X, y=None):
        pass

    def apply(self, X):
        return np.zeros(len(X)).astype(np.int)

    def __deepcopy__(self, memodict={}):
        return c_deepcopy(self)


class PiecewisePolynomialTree(BaseEstimator):
    """
    This is a simple PL-Tree which could be used for determine the best partition number
    """

    def __init__(self, min_samples_leaf=None, **param):
        self.label_model = defaultdict()
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y=None):
        self.tree_ = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf)
        self.tree_.fit(X, y)
        labels = self.tree_.apply(X)
        # X = PolynomialFeatures(degree=2).fit_transform(X)
        for l in np.unique(labels):
            index = labels == l
            lr = RidgeCV(np.logspace(0, 4))
            lr.fit(X[index], y[index])
            self.label_model[l] = lr

    def predict(self, X, y=None):
        # y_pred = np.zeros((X.shape[0], 1))
        y_pred = np.zeros((X.shape[0],))
        labels = self.tree_.apply(X)
        # X = PolynomialFeatures(degree=2).fit_transform(X)
        for l in self.label_model.keys():
            index = labels == l
            if np.sum(index) > 0:
                y_pred[index] = self.label_model[l].predict(X[index])
        return y_pred

    def apply(self, X):
        return np.zeros(len(X)).astype(np.int)

    def __deepcopy__(self, memodict={}):
        return copy.deepcopy(self)


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(builtins.max(aspirants, key=lambda x: np.sum(x.fitness.wvalues)))
    return chosen


def selTournamentDCD(individuals, k):
    """
    A simplified version of the tournament selection operator based on dominance and crowding distance

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """

    def tourn(ind1, ind2):
        if ind1.fitness.dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.dominates(ind1.fitness):
            return ind2

        if hasattr(ind1.fitness, 'crowding_dist') and hasattr(ind2.fitness, 'crowding_dist'):
            if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
                return ind2
            elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    chosen = []
    for i in range(0, k, 2):
        individuals_sample = random.sample(individuals, 4)
        chosen.append(tourn(individuals_sample[0], individuals_sample[1]))
        chosen.append(tourn(individuals_sample[2], individuals_sample[3]))
    return chosen


def automatically_determine_best_size(X, y, min_samples_leaf):
    """
    Automatically determine the best tree size
    """
    low_score = -np.inf
    best_size = 0
    for size in {
        'Auto': reversed([50, 100, 150, 200]),
        'Auto-4': reversed([25, 50, 100, 500]),
        'Auto-6': reversed([50, 75, 100, 125, 150, 200]),
        'Auto-8': reversed([25, 50, 75, 100, 125, 150, 200, 500]),
    }[min_samples_leaf]:
        dt = PiecewisePolynomialTree(min_samples_leaf=size)
        score = cross_validate(dt, X, y, scoring='neg_mean_squared_error', cv=5,
                               return_train_score=True)
        mean_score = np.mean(score['test_score'])
        if mean_score > low_score:
            # current score is better
            low_score = mean_score
            best_size = size
    return best_size
