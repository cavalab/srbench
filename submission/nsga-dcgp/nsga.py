import copy
import random
import signal

import torch
from sklearn.base import RegressorMixin, BaseEstimator
from .utils import accuracy, simplicity, dominate, print_info


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
            newton_step=10, stop=1e-6, verbose=None
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
        self.max_time = None

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
                    # here I igore the crowding distance
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



