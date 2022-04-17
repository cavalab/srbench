"""The underlying data structure used in taylorGP.

The :mod:`taylorGP._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`taylorGP.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause
from copy import copy
import numpy as np
from sklearn.utils.random import sample_without_replacement
from .functions import _Function,_sympol_map
from .utils import check_random_state
from taylorGP._global import set_value,get_value

def print_program(program,qualified_list,X,_x):
    n_features = X.shape[1]
    bias = qualified_list[-3]
    if qualified_list[-1] == 2:
        program = '-('+str(program) +')'
    partity = qualified_list[-2]
    if partity == 1 or partity == 2:
        if abs(bias)>1e-5:
            program = str(program) + '+('+str(bias) +')'
    __x = []
    for i in range(X.shape[1]):
        __x.append('x'+str(i))
    for i in range(X.shape[1]):
        program = program.replace(__x[i],str(_x[i]))
    print(program)
    return program
class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`taylorGP.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 selected_space = None,
                 qualified_list = None,
                 X = None,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 eq_write = None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.selected_space = selected_space
        self.qualified_list = qualified_list
        self.X =X
        self.eq_write = eq_write
        if self.program is not None :
            if not self.validate_program() :
                raise ValueError('The supplied program is incomplete.')
            while True:
                qiantao_flag = self.judge_qiantao(self)
                if qiantao_flag:
                    self.build(random_state,self.X)
                    break
                break
        else:
            self.build(random_state,self.X)



        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
    def build(self,random_state,X):
        build_flag = False
        qualified_flag = False
        qiantao_flag = False
        buildNum = 0
        while build_flag == False or qualified_flag == False:
            buildNum +=1
            if buildNum > 10000:
                set_value('TUIHUA_FLAG',True)
            self.program = self.build_program(random_state)
            qiantao_flag = self.judge_qiantao(self)
            if 'X' in str(self) and qiantao_flag==False:
                build_flag = True
                if get_value('TUIHUA_FLAG'):
                    break
            else:
                continue
            y_pred = self.execute(X)
            y_pred_reverse = self.execute(X * (-1))
            qualified_flag = self.isQualified(y_pred,y_pred_reverse)

    def isQualified(self,y_pred,y_pred_reverse):
        partity = False
        monotonicity =False
        if self.qualified_list[0] == -1:
            partity = True
        elif self.qualified_list[0] == self.judge_program_parity(y_pred,y_pred_reverse):
            partity = True
        if self.qualified_list[1] == -1:
            monotonicity = True
        elif self.qualified_list[1] == self.judge_program_monotonicity(y_pred):
            monotonicity = True
        if partity and monotonicity :
            return True
        else:
            return False


    def build_program(self, random_state):
        """Build a naive random program.
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)
        func_int_map = {}
        for i in range(len(self.function_set)):
            func_int_map[self.function_set[i].name] = i
        space = random_state.randint(len(self.selected_space))
        # space = -1
        space = self.selected_space[space]
        space = space.split()
        try:
            function1 = self.function_set[func_int_map[space[0]]]
        except BaseException:
            return [int(space[0])]
        program = [function1]
        terminal_stack = [function1.arity]
        right_subttree_flag = True
        if function1.arity == 1 :
            right_subttree_flag = False

        if space[1] in ['add','sub','mul','div','sin','cos','log','exp','sqrt']:
            function2 = self.function_set[func_int_map[space[1]]]
            program.append(function2)
            terminal_stack.append(function2.arity)
        else:
            if space[1] == 'const':
                terminal = random_state.uniform(*self.const_range)
            else:
                terminal = int(space[1])
            program.append(terminal)
            terminal_stack[-1] -= 1
            while terminal_stack[-1] == 0:
                terminal_stack.pop()
                if not terminal_stack:
                    return program
                terminal_stack[-1] -= 1


        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)


            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
            if len(terminal_stack) == 1 and right_subttree_flag:
                if space[2] in ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'exp', 'sqrt']:
                    function3 = self.function_set[func_int_map[space[2]]]
                    program.append(function3)
                    terminal_stack.append(function3.arity)
                else:
                    if space[2] == 'const':
                        terminal = random_state.uniform(*self.const_range)
                    else:
                        terminal = int(space[2])
                    program.append(terminal)
                    terminal_stack[-1] -= 1
                    while terminal_stack[-1] == 0:
                        terminal_stack.pop()
                        if not terminal_stack:
                            return program
                        terminal_stack[-1] -= 1
        # We should never get here
        return None

    def judge_qiantao(self,f):
        f = str(f)
        if 'zoo' in f or 'nan' in f or 'I' in f:
            return True
        tnum = 4
        Cexp = f.count('exp')
        index = 0

        for i in range(Cexp):
            index = f.find('exp', index)  # exp
            index = index + 4
            strNum = 1  # str = '('
            ind = index
            while strNum != 0:
                if f[ind] == 'e' and f[ind + 1] == 'x':
                    return True
                elif f[ind] == '(':
                    strNum += 1
                elif f[ind] == ')':
                    strNum -= 1
                ind += 1
        return False
    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]
    def get_expression(self):
        """return a sympy formula."""
        terminals = [0]
        output = ''
        stack = []
        X_num = [] #the variable of the sympy formula
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                if node.arity == 2:
                    if node.name not in 'addsubmuldiv':
                        output += node.name + '('
                    else:
                        output += '('
                    stack.append(node)
                elif node.arity == 1:
                    output += node.name + '('
                else:
                    print("node arity error!")
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'x%s' % node
                        if node not in X_num:
                            X_num.append(node)
                    else:
                        output += self.feature_names[node]
                else:
                    if node<0:
                        output += '(%.3f)' % node
                    else:
                        output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    if stack[-1].name in 'addsubmuldiv':
                        output += _sympol_map[f'{stack[-1].name}']
                        stack.pop()
        return output

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):#层次遍历顺序的terminal是错误的
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node,X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness
    def judge_program_parity(self,y_pred,y_pred_reverse):
        '''Judging the parity of randomly generated individuals！！！'''
        Jishu, Oushu = False, False
        y_1 = abs(y_pred + y_pred_reverse ) <0.01
        if True in y_1:
            Jishu = True
        y_2 = abs(y_pred - y_pred_reverse ) <0.01
        if True in y_2:
            Oushu = True
        if Jishu == True and Oushu == False:
            return 1
        elif Jishu == False and Oushu == True:
            return 2
        return -1

    def judge_program_monotonicity(self,y_pred):
        Increase,Decrease = False,False

        Y_index = np.argsort(y_pred, axis=0)
        Y_index = Y_index.reshape(-1)
        for i in range(1, Y_index.shape[0]):
            Increase_flag = not any([(self.X[Y_index[i]][j] < self.X[Y_index[i - 1]][j]) for j in range(self.X.shape[1])])
            if Increase_flag:
                Increase = True
            Decrease_flag = not any([(self.X[Y_index[i]][j] > self.X[Y_index[i - 1]][j]) for j in range(self.X.shape[1])])
            if Decrease_flag:
                Decrease = True

        if Increase == True and Decrease == False:
            return 1
        elif Increase == False and Decrease == True:
            return 2
        return -1

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign#sign控制正负号--越大越好还是越小越好
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())

        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state,qualified_list):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.
        donor program：
        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        qualified_flag = False
        op_index = 0
        while qualified_flag == False:
            op_index = random_state.randint(5)
            if get_value('TUIHUA_FLAG'):
                break
            elif qualified_list == [1,-1] and (op_index == 2 or op_index==3):
                continue
            elif qualified_list == [2, -1] and op_index == 4 and self.n_features>1:
                continue
            elif (qualified_list == [-1, 1] or qualified_list == [-1, 2]) and (op_index == 1 or op_index == 3):
                continue
            elif (qualified_list == [1, 1] or qualified_list == [-1, 2]) and (op_index == 1 or op_index == 2 or op_index == 3):
                continue
            qualified_flag = True


        if op_index <4 :
            program = self.function_set[op_index:op_index + 1] + self.program[:] + donor[:]
            return  program,None,None
        else:
            x_index = random_state.randint(self.n_features)
            if x_index not in self.program:
                for i in range(len(self.program)):
                    if isinstance(self.program[i],int):
                        x_index = self.program[i]
                        break

            for node in range(len(self.program)):
                if isinstance(self.program[node], _Function) == False and self.program[node] == x_index:
                    terminal = donor
                    program = self.changeTo(self.program, node, terminal)
            return program,None,None

    def changeTo(self,program,node, terminal):
        return program[:node] + terminal + program[node+1:]
    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
