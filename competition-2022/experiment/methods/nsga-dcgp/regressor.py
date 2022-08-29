import copy
import signal
from functools import partial, reduce
from random import randint
import random

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
from sklearn.base import BaseEstimator, RegressorMixin


MIN_SIM, MAX_SIM = 0., 1.


def reduce_adder(l1, l2):
    return l1 + l2


def simplicity(indiv):
    expr = indiv.expr()
    try:
        eq = sp.sympify(expr)
        s = eq.count_ops()
    except BaseException:
        s = len(expr)
    return round(30 / (30 + s), 1)


def accuracy(indiv, X, y):
    with torch.no_grad():
        y_pred = indiv(X)
    y_pred = y_pred.detach()
    sse = ((y_pred - y) ** 2).sum()
    var = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / var
    if not torch.isfinite(r2):
        return -np.inf
    return r2.item()


def dominate(obj_p, obj_q):
    # here, greater is better
    obj_p, obj_q = np.array(obj_p), np.array(obj_q)
    return (obj_p >= obj_q).all() and (obj_p > obj_q).any()


def crowding_distance(pop, obj_names):
    for i in range(len(pop)):
        pop[i].crowding_distance = 0
    for obj_name in obj_names:
        pop.sort(key=lambda indiv: getattr(indiv, obj_name))
        pop[0].crowding_distance = pop[-1].crowding_distance = np.inf
        for i in range(1, len(pop)-1):
            # since fitness is R2 score, which is range in (-inf, 1]
            # so I ignore normalization here
            pop[i].crowding_distance += getattr(pop[i+1], obj_name) - getattr(pop[i-1], obj_name)


def print_info(gen, indiv):
    print("Gen {}: Best EQ={} || R2Score={}, Simplicity={}".format(gen, indiv.expr(), indiv.fitness, indiv.simplicity))


class Function:
    def __init__(self, pt_func, sp_func, arity):
        self.pt_func = pt_func
        self.sp_func = sp_func
        self.arity = arity

    def __call__(self, *args):
        return self.pt_func(*args)

    def expr(self, *args):
        return self.sp_func(*args)


def _sp_add(*args):
    return f"Add({args[0]}, {args[1]})"


def _sp_sub(*args):
    return f"Add({args[0]}, -{args[1]})"


def _sp_mul(*args):
    return f"Mul({args[0]}, {args[1]})"


def _sp_div(*args):
    return f"Mul({args[0]}, 1/{args[1]})"


def _unary_sp_oper(*args, op=''):
    return f"{op}({args[0]})"


def _sp_log(*args):
    return f"log(Abs({args[0]}))"


def _sp_sqrt(*args):
    return f"sqrt(Abs({args[0]}))"


def _sp_power(*args, base=None):
    if base:
        return f"({args[0]})**{str(base)}"
    return f"{args[0]}"


def _protected_log(*args):
    return torch.log(torch.abs(args[0]))


def _protected_sqrt(*args):
    return torch.sqrt(torch.abs(args[0]))


sqrt = Function(_protected_sqrt, _sp_sqrt, 1)
square = Function(torch.square, partial(_sp_power, base=2), 1)
cube = Function(partial(torch.pow, exponent=3), partial(_sp_power, base=3), 1)
log = Function(_protected_log, _sp_log, 1)
sin = Function(torch.sin, partial(_unary_sp_oper, op='sin'), 1)
cos = Function(torch.cos, partial(_unary_sp_oper, op='cos'), 1)
# tan = Function(torch.tan, sp.tan, 1)
abs = Function(torch.abs, partial(_unary_sp_oper, op='Abs'), 1)
# exp = Function(torch.exp, sp.exp, 1)

add = Function(torch.add, _sp_add, 2)
sub = Function(torch.sub, _sp_sub, 2)
mul = Function(torch.mul, _sp_mul, 2)
div = Function(torch.div, _sp_div, 2)

primitive_map = {
    'sqrt': sqrt,
    'square': square,
    'cube': cube,
    'log': log,
    'sin': sin,
    'cos': cos,
    # 'tan': tan,
    'abs': abs,
    # 'exp': exp,

    'add': add,
    'sub': sub,
    'mul': mul,
    'div': div
}

default_primitives = list(primitive_map.keys())


def create_functions(primitives):
    if isinstance(primitives[0], Function):
        return primitives
    return list(primitive_map[primitive] for primitive in primitives)


def max_arity(functions):
    ma = 0
    for func in functions:
       ma = max(func.arity, ma)
    return ma


class Parameter:
    def __init__(
            self,
            n_var=None, n_output=1,
            n_row=10, n_col=10, n_constant=1,
            primitive_set=None,
            levels_back=None
    ):
        self.n_var = n_var
        self.n_output = n_output
        self.n_constant = n_constant
        self.n_row = n_row
        self.n_col = n_col
        if primitive_set is None:
            primitive_set = default_primitives
        self.function_set = create_functions(primitive_set)
        self.max_arity = max_arity(self.function_set)
        self.levels_back = self.n_col if levels_back is None else levels_back


class Node:
    def __init__(self, id, func, inputs, arity=0, is_input=False, is_output=False):
        self.id = id
        self.func = func
        self.inputs = inputs
        self.arity = arity
        self.is_input = is_input
        self.is_output = is_output

        self.value = None


class DifferentialCGP(torch.nn.Module):
    def __init__(
            self, hyper_param: Parameter,
            genes=None, bounds=None, constant=None
    ):
        super(DifferentialCGP, self).__init__()
        self.hyper_param = hyper_param
        # unpack for convinence
        self.n_var = hyper_param.n_var
        self.n_constant = hyper_param.n_constant
        self.n_input = self.n_var + self.n_constant
        self.n_output = hyper_param.n_output
        self.n_row = hyper_param.n_row
        self.n_col = hyper_param.n_col
        self.function_set = hyper_param.function_set
        self.max_arity = hyper_param.max_arity
        self.levels_back = hyper_param.levels_back

        if genes is None:
            self.genes, self.bounds = self.initialization()
            self.constant = torch.nn.Parameter(torch.normal(mean=0., std=1., size=(self.n_constant,)), requires_grad=True)
        else:
            self.genes, self.bounds = genes, bounds
            self.constant = constant if isinstance(constant, torch.nn.Parameter) else torch.nn.Parameter(constant, requires_grad=True)
        self.nodes = self._create_nodes(self.genes)
        self.active_paths = self._active_paths(self.nodes)
        self.active_nodes = set(reduce(reduce_adder, self.active_paths))

        self.fitness = None
        self.simplicity = None
        self.front_rank = None
        self.dominated_set = None
        self.n_dominator = None
        self.crowding_distance = None

    def _active_paths(self, nodes):
        stack = []
        active_path, active_paths = [], []
        for node in reversed(nodes):
            if node.is_output:
                stack.append(node)
            else:
                break

        while len(stack) > 0:
            node = stack.pop()

            if len(active_path) > 0 and node.is_output:
                active_paths.append(list(reversed(active_path)))
                active_path = []

            active_path.append(node.id)

            for input_gene in node.inputs:
                stack.append(nodes[input_gene])

        if len(active_path) > 0:
            active_paths.append(list(reversed(active_path)))

        return active_paths

    def _create_nodes(self, genes):
        nodes = []
        for i in range(self.n_input):
            nodes.append(Node(i, None, [], is_input=True))

        f_pos = 0
        for i in range(self.n_row * self.n_col):
            func = self.function_set[genes[f_pos]]
            input_genes = genes[f_pos + 1: f_pos + 1 + func.arity]
            nodes.append(Node(i + self.n_input, func, input_genes, arity=func.arity))
            f_pos += self.max_arity + 1

        end_func_node = self.n_input + self.n_row * self.n_col
        for gene in genes[-self.n_output:]:
            nodes.append(Node(end_func_node, None, [gene], is_output=True))
            end_func_node += 1

        return nodes

    def initialization(self):
        genes, bounds = [], []
        for col in range(self.n_col):
            for row in range(self.n_row):
                # func node
                function_gene = randint(0, len(self.function_set) - 1)
                genes.append(function_gene)
                bounds.append((0, len(self.function_set) - 1))

                # input gene
                upper = col * self.n_row + self.n_input - 1
                lower = max(0, upper + 1 - self.levels_back * self.n_row - 1)
                for input_idx in range(self.max_arity):
                    bounds.append((lower, upper))
                    genes.append(randint(lower, upper))

        # output gene
        upper = self.n_col * self.n_row + self.n_input - 1
        lower = max(0, upper + 1 - self.levels_back * self.n_row - 1)
        for i in range(self.n_output):
            bounds.append((lower, upper))
            genes.append(randint(lower, upper))

        return genes, bounds

    def forward(self, X):
        for path in self.active_paths:
            for gene in path:
                node = self.nodes[gene]
                if node.is_input:
                    node.value = self.constant[node.id - self.n_input] if node.id >= self.n_var else X[:, node.id]
                elif node.is_output:
                    node.value = self.nodes[node.inputs[0]].value
                else:
                    f = node.func
                    operants = [self.nodes[node.inputs[i]].value for i in range(node.arity)]
                    node.value = f(*operants)

        if self.n_output == 1:
            if len(self.nodes[-1].value.shape) == 0:
                self.nodes[-1].value = self.nodes[-1].value.repeat(X.shape[0])
            return self.nodes[-1].value

        outputs = []
        for node in self.nodes[-self.n_output:]:
            if len(node.value.shape) == 0:
                outputs.append(node.value.repeat(X.shape[0]))
            else:
                outputs.append(node.value)

        return torch.stack(outputs, dim=1)

    def expr(self, variables=None):
        if variables is None:
            variables = list(['x{}'.format(i) for i in range(self.n_var)])
        symbol_stack = []
        results = []
        for path in self.active_paths:
            for i_node in path:
                node = self.nodes[i_node]
                if node.is_input:
                    if i_node >= self.n_var:
                        c = self.constant[i_node - self.n_var].item()
                    else:
                        c = variables[i_node]
                    symbol_stack.append(c)
                elif node.is_output:
                    results.append(symbol_stack.pop())
                else:
                    f = node.func
                    # get a sympy symbolic expression.
                    symbol_stack.append(f.expr(*reversed([symbol_stack.pop() for _ in range(f.arity)])))

        return results[0] if len(results) == 1 else results

    def mutate(self, probability):
        new_genes = self.genes[:]
        bounds = self.bounds[:]

        for gidx in range(len(new_genes)):
            chance = random.random()
            if chance < probability:
                low, up = bounds[gidx]
                candicates = [g for g in range(low, up+1) if g != self.genes[gidx]]
                if len(candicates) == 0:
                    continue
                new_genes[gidx] = random.choice(candicates)

        return DifferentialCGP(
            self.hyper_param,
            genes=new_genes, bounds=bounds, constant=self.constant
        )


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    print('raising TimeOutException')
    raise TimeOutException


class NSGA(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            indiv_class, indiv_param,
            pop_size=100, n_gen=10000, n_parent=15, prob=0.4, nsga=True,
            newton_step=10, stop=1e-6, verbose=None,
            max_time=None
    ):
        self.indiv_class = indiv_class
        self.indiv_param = indiv_param
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_parent, self.n_offspring = n_parent, self.pop_size - n_parent
        self.prob = prob
        self.nsga = nsga
        self.newton_step = newton_step
        self.stop = stop
        self.parent, self.best_solution, self.fronts = None, None, None
        self.verbose = verbose
        self.loss_func = torch.nn.MSELoss()
        self.max_time = max_time

    def set_max_time(self, max_time):
        self.max_time = max_time

    def _init_pop(self):
        self.parent = list([self.indiv_class(self.indiv_param) for _ in range(self.n_parent)])
        return self._ea()

    def _optim_constant(self, X, y, pop):
        for i in range(len(pop)):
            old_constant = torch.FloatTensor(pop[i].constant.size()).type_as(pop[i].constant)
            for step in range(self.newton_step):
                y_prediction = pop[i](X)
                loss = self.loss_func(y_prediction, y)
                if loss.requires_grad:
                    grad_c = torch.autograd.grad(loss, pop[i].constant, create_graph=True)
                    hessian = torch.autograd.grad(grad_c[0], pop[i].constant)
                    pop[i].constant.data -= grad_c[0].data / hessian[0].data
                else:
                    break
            if not torch.isfinite(pop[i].constant).all():
                pop[i].constant.data = old_constant

    def _ea(self):
        # E(u+lambda)
        offspring = list([random.choice(self.parent).mutate(self.prob) for _ in range(self.n_offspring)])
        return self.parent + offspring

    def _fast_nondoiminated_sort(self, population):
        # implementing fatst-NS sorting according to [K Deb, A Pratap, S Agarwal et.al 2002]
        fronts, pareto_front = [], []
        for p in population:
            Sp, Np = [], 0
            for q in population:
                obj_p, obj_q = [p.fitness, p.simplicity], [q.fitness, q.simplicity]
                if dominate(obj_p, obj_q):
                    Sp.append(q)
                elif dominate(obj_q, obj_p):
                    Np += 1
            if Np == 0:
                p.front_rank = 0
                pareto_front.append(p)
            p.dominated_set = Sp
            p.n_dominator = Np

        fronts.append(pareto_front)
        i = 0
        while len(fronts[i]) != 0:
            Q = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.n_dominator -= 1
                    if q.n_dominator == 0:
                        q.front_rank = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)

        return fronts

    def fit(self, X, y):
        if self.max_time is not None:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(self.max_time)
        n_var = X.shape[1]
        self.indiv_param.n_var = n_var
        # X_train, X_val, y_train, y_val = train_test_split(X.values, y)
        input_tensor = torch.from_numpy(X).float()
        target_tensor = torch.from_numpy(y).float()
        # initialization
        population = self._init_pop()
        try:
            for gen in range(self.n_gen):
                # optimizer constant
                self._optim_constant(input_tensor, target_tensor, population)
                # calculate fitness and simplicity
                for i in range(len(population)):
                    population[i].fitness = accuracy(population[i], input_tensor, target_tensor)
                    population[i].simplicity = simplicity(population[i])
                if self.nsga:
                    # fast non-dominated sorting
                    self.fronts = self._fast_nondoiminated_sort(population)
                    # plot_pareto(population)
                    # select u parent from pareto-fronts
                    new_parent, i = [], 0
                    while len(new_parent) + len(self.fronts[i]) <= self.n_parent:
                        new_parent += self.fronts[i]
                        i += 1
                    # non-normalized crowding distance
                    crowding_distance(self.fronts[i], ['fitness', 'simplicity'])
                    # prefer the less crowded region (i.e. crowding_distance is larger)
                    self.fronts[i].sort(key=lambda indiv: indiv.crowding_distance, reverse=True)
                    new_parent += self.fronts[i][:self.n_parent - len(new_parent)]
                    self.parent = new_parent
                else:
                    self.parent = sorted(population, key=lambda indiv: indiv.fitness)[-self.n_parent:]
                # E(u+lambda) evolutionary strategy
                population = self._ea()

                if self.best_solution is None:
                    self.best_solution = copy.deepcopy(max(self.parent, key=lambda indiv: indiv.fitness))
                else:
                    new_best = max(self.parent, key=lambda indiv: indiv.fitness)
                    if new_best.fitness > self.best_solution.fitness:
                        self.best_solution = copy.deepcopy(new_best)
                if self.verbose is not None and gen % self.verbose == 0:
                    print_info(gen, self.best_solution)
                if (1. - self.best_solution.fitness) <= self.stop:
                    break
        except TimeOutException:
            if self.best_solution is None:
                self.best_solution = max(population if self.parent is None else self.parent, key=lambda indiv: indiv.fitness)

    def predict(self, X):
        assert self.best_solution is not None, "Never call fit() before"
        input_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            prediction_tensor = self.best_solution(input_tensor)
        return prediction_tensor.detach().numpy()

    def expr(self, variables=None):
        assert self.best_solution is not None, "Never call fit() before"
        return self.best_solution.expr(variables)


dcgp_params = Parameter(
    n_output=1,
    n_row=10, n_col=10, n_constant=1,
    primitive_set=None,
    levels_back=None
)
est = NSGA(
    DifferentialCGP, dcgp_params,
    pop_size=1000, n_gen=10000, n_parent=200, prob=0.4, nsga=True,
    newton_step=10, stop=1e-6, verbose=None
)


def model(est, X=None):
    # sympify it first
    model_ = est.expr()
    if X is None or not hasattr(X, 'columns'):
        return model_

    mappings = {'x' + str(i): k for i, k in enumerate(X.columns)}
    for k, v in reversed(mappings.items()):
        model_ = model_.replace(k, v)
    return model_

################################################################################
# Optional Settings
################################################################################


"""
eval_kwargs: a dictionary of variables passed to the evaluate_model()
    function. 
    Allows one to configure aspects of the training process.

Options 
-------
    test_params: dict, default = None
        Used primarily to shorten run-times during testing. 
        for running the tests. called as 
            est = est.set_params(**test_params)
    max_train_samples:int, default = 0
        if training size is larger than this, sample it. 
        if 0, use all training samples for fit. 
    scale_x: bool, default = True 
        Normalize the input data prior to fit. 
    scale_y: bool, default = True 
        Normalize the input label prior to fit. 
    pre_train: function, default = None
        Adjust settings based on training data. Called prior to est.fit. 
        The function signature should be (est, X, y). 
            est: sklearn regressor; the fitted model. 
            X: pd.DataFrame; the training data. 
            y: training labels.
"""


def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X 
       versus relative to FEAT's batch size setting. 
    """
    if len(X) <= 1000:
        max_time = 3600 - 10
    else:
        max_time = 36000 - 10
    est.set_params(max_time=max_time)


# define eval_kwargs.
eval_kwargs = dict(
    pre_train=my_pre_train_fn,
    test_params={
        "pop_size": 10,
        "n_gen": 10,
        "n_parent": 2
    },
    DataFrame=False
)


# # test here
# test_parms = {
#     "pop_size": 100,
#     "n_gen": 100,
#     "n_parent": 16,
#     "verbose": 1
# }
# dataset_path = '/home/luoyuanzhen/STORAGE/dataset/sr_benchmark/Korns-3_train.txt'
# dataset = np.loadtxt(dataset_path)
# X, y = dataset[:, :-1], dataset[:, -1]
#
#
# def true_model(x):
#     return (-x[:, 0] + x[:, 3] + x[:, 3]/x[:, 4])/x[:, 4]
#
#
# predict = true_model(X)
# from sklearn.metrics import r2_score
# print(r2_score(predict, y))
# exit()
#
# my_pre_train_fn(est, X, y)
# est.set_params(**test_parms)
#
# est.fit(X, y)
# str_eq = model(est, X)
# print(str_eq)
# print(sp.sympify(str_eq))
