import random
from functools import reduce
from random import randint

import torch.nn

from primitives import default_primitives, create_functions, max_arity
from utils import reduce_adder


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

            for input_gene in reversed(node.inputs):
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
