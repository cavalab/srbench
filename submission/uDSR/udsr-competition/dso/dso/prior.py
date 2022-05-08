"""Class for Prior object."""
import os
import warnings
import inspect
import copy
from collections import defaultdict
from copy import deepcopy

import numpy as np
from prettytable import PrettyTable

from dso.library import TokenNotFoundError, Token, StateChecker
from dso.subroutines import ancestors
from dso.subroutines import jit_check_constraint_violation, \
        jit_check_constraint_violation_descendant_with_target_tokens, \
        jit_check_constraint_violation_descendant_no_target_tokens, \
        jit_check_constraint_violation_uchild, get_position, get_mask
from dso.program import Program
from dso.language_model import LanguageModelPrior as LM
import dso.constants as constants
from dso.utils import import_custom_source
import dso.task.binding.utils as BU


def make_prior(library, config_prior):
    """Factory function for JointPrior object."""

    config_prior = deepcopy(config_prior)

    prior_dict = {
        "relational" : RelationalConstraint,
        "length" : LengthConstraint,
        "repeat" : RepeatConstraint,
        "inverse" : InverseUnaryConstraint,
        "trig" : TrigConstraint,
        "const" : ConstConstraint,
        "no_inputs" : NoInputsConstraint,
        "soft_length" : SoftLengthPrior,
        "uniform_arity" : UniformArityPrior,
        "domain_range" : DomainRangeConstraint,
        "seq_positions": SequencePositionsConstraint,
        "language_model" : LanguageModelPrior,
        "abag_seq_model": ABAGSeqPrior,
        "mut_dist" : MutationalDistanceConstraint,
        "mut_prob" : MutationProbabilityPrior
    }

    count_constraints = config_prior.pop("count_constraints", False)

    priors = []
    warn_messages = []
    for prior_type, prior_args in config_prior.items():
        if prior_type in prior_dict:
            prior_class = prior_dict[prior_type]
        else:
            # Tries to import custom priors
            prior_class = import_custom_source(prior_type)

        if isinstance(prior_args, dict):
            prior_args = [prior_args]
        for single_prior_args in prior_args:
            # Attempt to build the Prior. Any Prior can fail if it references a
            # Token not in the Library.
            prior_is_enabled = single_prior_args.pop('on', False)
            if prior_is_enabled:
                try:
                    prior = prior_class(library, **single_prior_args)
                    warn_message = prior.validate()
                except TokenNotFoundError:
                    prior = None
                    warn_message = "Uses Tokens not in the Library."
            else:
                prior = None
                warn_message = "Prior disabled."

            # Add warning context
            if warn_message is not None:
                warn_message = "Skipping invalid '{}' with arguments {}. " \
                    "Reason: {}" \
                    .format(prior_class.__name__, single_prior_args, warn_message)
                warn_messages.append(warn_message)

            # Add the Prior if there are no warnings
            if warn_message is None:
                priors.append(prior)

    # Turn PolyConstraint "on" if and only if "poly" token is in library
    if "poly" in library.names:
        priors.append(PolyConstraint(library))

    # Turn StateCheckerConstraint "on" if and only if StateChecker is in library
    if len(library.state_checker_tokens) > 0:
        priors.append(StateCheckerConstraint(library))

    joint_prior = JointPrior(library, priors, count_constraints)

    print("-- BUILDING PRIOR START -------------")
    print("\n".join(["WARNING: " + message for message in warn_messages]))
    print(joint_prior.describe())
    print("-- BUILDING PRIOR END ---------------\n")

    return joint_prior


class JointPrior():
    """A collection of joint Priors."""

    def __init__(self, library, priors, count_constraints=False):
        """
        Parameters
        ----------
        library : Library
            The Library associated with the Priors.

        priors : list of Prior
            The individual Priors to be joined.

        count_constraints : bool
            Whether to count the number of constrained tokens.
        """

        self.library = library
        self.L = self.library.L
        self.priors = priors
        assert all([prior.library is library for prior in priors]), \
            "All Libraries must be identical."

        # Name the priors, e.g. RepeatConstraint-0
        counts = defaultdict(lambda : -1)
        self.names = []
        for prior in self.priors:
            name = prior.__class__.__name__
            counts[name] += 1
            self.names.append("{}-{}".format(name, counts[name]))

        # Initialize variables for constraint count report
        self.do_count = count_constraints
        self.constraint_counts = [0] * len(self.priors)
        self.total_constraints = 0
        self.total_tokens = 0

        self.requires_parents_siblings = True # TBD: Determine

        # SPECIAL CASE: Set DomainRangeConstraint.max_length if constraining max length
        length_prior = None
        domain_range_prior = None
        for prior in self.priors:
            if isinstance(prior, LengthConstraint):
                length_prior = prior
            elif isinstance(prior, DomainRangeConstraint):
                domain_range_prior = prior
        if length_prior is not None and domain_range_prior is not None:
            domain_range_prior.max_length = length_prior.max

        self.describe()

    def initial_prior(self):
        combined_prior = np.zeros((self.L,), dtype=np.float32)
        for prior in self.priors:
            combined_prior += prior.initial_prior()
        return combined_prior

    def __call__(self, actions, parent, sibling, dangling, finished):

        # Filter out finished sequences to save compute
        unfinished = ~finished
        final_combined_prior = np.zeros((actions.shape[0], self.L), dtype=np.float32)
        unfinished_actions = actions[unfinished]
        unfinished_parent = parent[unfinished]
        unfinished_sibling = sibling[unfinished]
        unfinished_dangling = dangling[unfinished]

        # Sum the individual priors
        zero_prior = np.zeros((unfinished_actions.shape[0], self.L), dtype=np.float32)
        ind_priors = [zero_prior.copy() for _ in range(len(self.priors))]
        for i in range(len(self.priors)):
            ind_priors[i] += self.priors[i](unfinished_actions, unfinished_parent, unfinished_sibling, unfinished_dangling)
        combined_prior = sum(ind_priors) + zero_prior # TBD FIX HACK

        # Count number of constrained tokens per prior
        if self.do_count:
            self.total_tokens += unfinished_actions.shape[0] * self.library.L
            for i in range(len(self.constraint_counts)):
                self.constraint_counts[i] += np.count_nonzero(ind_priors[i] == -np.inf)
            self.total_constraints += np.count_nonzero(combined_prior == -np.inf)

        # Give status report if a prior contains all -inf
        collision_mask = np.all(np.isneginf(combined_prior), axis=1)
        if np.any(collision_mask):
            collisions = collision_mask.nonzero()[0] # Indices of collisions
            msg = []
            msg.append("ERROR in {}:".format(__file__))
            msg.append("Encountered collision(s) in prior. This occurs when a prior contains all -inf values. " +
                       "This typically indicates a logic error in a prior formulation, or a 'collision' when " +
                       "configuring multiple priors that are not always compatible. See the table(s) below " +
                       "for which priors caused each collision. X's indicate a value of -inf.")
            for i, collision in enumerate(collisions):
                msg.append("\n----- Collision {} of {} -----".format(i + 1, len(collisions)))
                msg.append("Traversal: {}".format(self.library.tokenize(unfinished_actions[collision])))
                table = PrettyTable(["Prior"] + self.library.names)
                for j, ind_prior in enumerate(ind_priors):
                    if np.any(np.isneginf(ind_prior[collision])):
                        name = self.priors[j].__class__.__name__
                        row = ["X" if x == -np.inf else "" for x in ind_prior[collision]]
                        table.add_row([name] + row)
                msg.append(repr(table))
            # msg.append("\nApplication will now exit.")
            print("\n".join(msg))
            # os._exit(1) # Bypass tensorflow exception-handling

        final_combined_prior[unfinished] = combined_prior

        return final_combined_prior

    def report_constraint_counts(self):
        if not self.do_count:
            return
        print("Constraint counts per prior:")
        for i, count in enumerate(self.constraint_counts):
            print("{}: {} ({:%})".format(self.names[i], count, count / self.total_tokens))
        print("All priors: {} ({:%})".format(self.total_constraints, self.total_constraints / self.total_tokens))

    def describe(self):
        message = "\n".join(prior.describe() for prior in self.priors)
        return message

    def is_violated(self, actions, parent, sibling):
        for prior in self.priors:
            if isinstance(prior, Constraint):
                if prior.is_violated(actions, parent, sibling):
                    return True
        return False

    def at_once(self, actions, parent, sibling):
        """
        Given a full sequence of actions, parents, and siblings, each of shape
        (batch, time), *retrospectively* compute what was the joint prior at all
        time steps. The combined prior has shape (batch, time, L).
        """

        B, T = actions.shape
        zero_prior = np.zeros((B, T, self.L), dtype=np.float32) # (batch, time, L)
        ind_priors = [zero_prior.copy() for _ in range(len(self.priors))] # i x (batch, time, L)

        if len(self.priors) == 0:
            return zero_prior

        # Set initial prior
        # Note: intial_prior() is already a combined prior, so we just set the
        # first individual prior, ind_priors[0].
        initial_prior = self.initial_prior() # Shape (L,)
        ind_priors[0][:, 0, :] = initial_prior # Broadcast to (batch, L)

        dangling = np.ones(B)
        for t in range(1, T): # For each time step
            # Update dangling based on previously sampled token
            dangling += self.library.arities[actions[:, (t - 1)]] - 1
            for i in range(len(self.priors)): # For each Prior
                # Compute the ith Prior at time step t
                prior = self.priors[i](actions[:, :t],
                                       parent[:, t],
                                       sibling[:, t],
                                       dangling) # Shape (batch, L)
                ind_priors[i][:, t, :] += prior

        # Combine all Priors
        combined_prior = sum(ind_priors) + zero_prior

        return combined_prior


class Prior():
    """Abstract class whose call method return logits."""

    def __init__(self, library):
        self.library = library
        self.L = library.L
        self.mask_val = -np.inf

    def validate(self):
        """
        Determine whether the Prior has a valid configuration. This is useful
        when other algorithmic parameters may render the Prior degenerate. For
        example, having a TrigConstraint with no trig Tokens.

        Returns
        -------
        message : str or None
            Error message if Prior is invalid, or None if it is valid.
        """

        return None

    def init_zeros(self, actions):
        """Helper function to generate a starting prior of zeros."""

        batch_size = actions.shape[0]
        prior = np.zeros((batch_size, self.L), dtype=np.float32)
        return prior

    def initial_prior(self):
        """
        Compute the initial prior, before any actions are selected.

        Returns
        -------
        initial_prior : array
            Initial logit adjustment before actions are selected. Shape is
            (self.L,) as it will be broadcast to batch size later.
        """

        return np.zeros((self.L,), dtype=np.float32)

    def __call__(self, actions, parent, sibling, dangling):
        """
        Compute the prior (logit adjustment) given the history of actions
        and the current observation.

        actions : np.ndarray (dtype=np.int32)
            History of all actions so far. Each action is a Library index.
            Shape is (batch_size, current_time).

        parent : np.ndarray (dtype=np.in32)
            Adjusted Library indices of parent of the token being selected.
            Shape is (batch_size,).

        sibling : np.ndarray (dtype=np.int32)
            Library indices of sibling of the token being selected.
            Shape is (batch_size,).

        dangling : np.ndarray (dtype=np.int32)
            Current number of dangling/unselected nodes.
            Shape is (batch_size,).

        Returns
        -------
        prior : np.ndarray (dtype=np.float32)
            Logit adjustment for selecting next action.
            Shape is (batch_size, self.L).
        """

        raise NotImplementedError
        
    def describe(self):
        """Describe the Prior."""

        return "{}: No description available.".format(self.__class__.__name__)


class Constraint(Prior):
    def __init__(self, library):
        Prior.__init__(self, library)
        
    def make_constraint(self, mask, tokens):
        """
        Generate the prior for a batch of constraints and the corresponding
        Tokens to constrain.

        For example, with L=5 and tokens=[1,2], a constrained row of the prior
        will be: [0.0, -np.inf, -np.inf, 0.0, 0.0].

        Parameters
        __________

        mask : np.ndarray, shape=(?,), dtype=np.bool_
            Boolean mask of samples to constrain.

        tokens : np.ndarray, dtype=np.int32
            Tokens to constrain.

        Returns
        _______

        prior : np.ndarray, shape=(?, L), dtype=np.float32
            Logit adjustment. Since these are hard constraints, each element is
            either 0.0 or -np.inf.
        """
        prior = np.zeros((mask.shape[0], self.L), dtype=np.float32)
        
        for t in tokens:
            prior[mask, t] = self.mask_val
        return prior
    
    def is_violated(self, actions, parent, sibling):
        """
        Given a set of actions, tells us if a prior constraint has been violated 
        post hoc. 
        
        This is a generic version that will run using the __call__ function so that one
        does not have to write a function twice for both DSO and Deap. 
        
        >>>HOWEVER<<<
        
        Using this function is less optimal than writing a variant for Deap. So...
        
        If you create a constraint and find you use if often with Deap, you should gp ahead anf
        write the optimal version. 

        Returns
        -------
        violated : Bool
        """

        caller          = inspect.getframeinfo(inspect.stack()[1][0])

        warnings.warn("{} ({}) {} : Using a slower version of constraint for Deap. You should write your own.".format(caller.filename, caller.lineno, type(self).__name__))

        assert len(actions.shape) == 2, "Only takes in one action at a time since this is how Deap will use it."
        assert actions.shape[0] == 1, "Only takes in one action at a time since this is how Deap will use it."
        dangling        = np.ones((1), dtype=np.int32)
        
        # For each step in time, get the prior                                
        for t in range(actions.shape[1]):
            dangling    += self.library.arities[actions[:,t]] - 1   
            priors      = self.__call__(actions[:,:t], parent[:,t], sibling[:,t], dangling)
            # Does our action conflict with this prior?
            if priors[0,actions[0,t]] == -np.inf:
                return True

        return False
    
    def test_is_violated(self, actions, parent, sibling):
        r"""
            This allows one to call the generic version of "is_violated" for testing purposes
            from the derived classes even if they have an optimized version. 
        """
        return Constraint.is_violated(self, actions, parent, sibling)
    

class RelationalConstraint(Constraint):
    """
    Class that constrains the following:

        Constrain (any of) `targets` from being the `relationship` of (any of)
        `effectors`.

    Parameters
    ----------
    targets : list of Tokens
        List of Tokens, all of which will be constrained if any of effectors
        are the given relationship.

    effectors : list of Tokens
        List of Tokens, any of which will cause all targets to be constrained
        if they are the given relationship.

    relationship : choice of ["child", "descendant", "sibling", "uchild"]
        The type of relationship to constrain.
    """

    def __init__(self, library, targets, effectors, relationship):
        Prior.__init__(self, library)
        self.targets = library.actionize(targets)
        self.effectors = library.actionize(effectors)
        self.relationship = relationship

    def __call__(self, actions, parent, sibling, dangling):

        if self.relationship == "descendant":
            mask = ancestors(actions=actions,
                             arities=self.library.arities,
                             ancestor_tokens=self.effectors)
            prior = self.make_constraint(mask, self.targets)

        elif self.relationship == "child":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            mask = np.isin(parent, adj_parents)
            prior = self.make_constraint(mask, self.targets)

        elif self.relationship == "sibling":
            # The sibling relationship is reflexive: if A is a sibling of B,
            # then B is also a sibling of A. Thus, we combine two priors, where
            # targets and effectors are swapped.
            mask = np.isin(sibling, self.effectors)
            prior = self.make_constraint(mask, self.targets)
            mask = np.isin(sibling, self.targets)
            prior += self.make_constraint(mask, self.effectors)

        elif self.relationship == "uchild":
            # Case 1: parent is a unary effector
            unary_effectors = np.intersect1d(self.effectors,
                                             self.library.unary_tokens)
            adj_unary_effectors = self.library.parent_adjust[unary_effectors]
            mask = np.isin(parent, adj_unary_effectors)
            # Case 2: sibling is a target and parent is an effector
            adj_effectors = self.library.parent_adjust[self.effectors]
            mask += np.logical_and(np.isin(sibling, self.targets),
                                   np.isin(parent, adj_effectors))
            prior = self.make_constraint(mask, [self.targets])

        elif self.relationship == "lchild":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            mask = np.logical_and(np.isin(parent, adj_parents), np.equal(sibling, self.L))
            prior = self.make_constraint(mask, self.targets)

        elif self.relationship == "rchild":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            mask = np.logical_and(np.isin(parent, adj_parents), np.less(sibling, self.L))
            prior = self.make_constraint(mask, self.targets)

        return prior

    def is_violated(self, actions, parent, sibling):

        if self.relationship == "descendant":
            violated = jit_check_constraint_violation_descendant_with_target_tokens(
                actions, self.targets, self.effectors, self.library.binary_tokens, self.library.unary_tokens)

        elif self.relationship == "child":
            parents = self.effectors
            adj_parents = self.library.parent_adjust[parents]
            violated = jit_check_constraint_violation(actions, self.targets, parent, adj_parents)

        elif self.relationship == "sibling":
            violated = jit_check_constraint_violation(actions, self.targets, sibling, self.effectors)
            if not violated:
                violated = jit_check_constraint_violation(actions, self.effectors, sibling, self.targets)

        elif self.relationship == "uchild":
            unary_effectors = np.intersect1d(self.effectors, self.library.unary_tokens)
            adj_unary_effectors = self.library.parent_adjust[unary_effectors]
            adj_effectors = self.library.parent_adjust[self.effectors]
            violated = jit_check_constraint_violation_uchild(actions, parent, sibling, self.targets, 
                                                     adj_unary_effectors, adj_effectors)

        return violated

    def validate(self):
        message = []
        if self.relationship in ["child", "descendant", "uchild", "lchild", "rchild"]:
            if np.isin(self.effectors, self.library.terminal_tokens).any():
                message = "{} relationship cannot have terminal effectors." \
                          .format(self.relationship.capitalize())
                return message
        if len(self.targets) == 0:
            message = "There are no target Tokens."
            return message
        if len(self.effectors) == 0:
            message = "There are no effector Tokens."
            return message
        return None

    def describe(self):

        targets = ", ".join([self.library.names[t] for t in self.targets])
        effectors = ", ".join([self.library.names[t] for t in self.effectors])
        relationship = {
            "child" : "a child",
            "sibling" : "a sibling",
            "descendant" : "a descendant",
            "uchild" : "the only unique child",
            "lchild" : "the left child",
            "rchild" : "the right child"
        }[self.relationship]
        message = "{}: [{}] cannot be {} of [{}]." \
                  .format(self.__class__.__name__, targets, relationship, effectors)
        return message


class TrigConstraint(RelationalConstraint):
    """Class that constrains trig Tokens from being the descendants of trig
    Tokens."""

    def __init__(self, library):
        targets = library.trig_tokens
        effectors = library.trig_tokens
        super(TrigConstraint, self).__init__(library=library,
                                             targets=targets,
                                             effectors=effectors,
                                             relationship="descendant")
        
    def is_violated(self, actions, parent, sibling):
        
        # Call a slightly faster descendant computation since target is the same as effectors
        return jit_check_constraint_violation_descendant_no_target_tokens(\
                actions, self.effectors, self.library.binary_tokens, self.library.unary_tokens)


class ConstConstraint(RelationalConstraint):
    """Class that constrains the const Token from being the only unique child
    of all non-terminal Tokens."""

    def __init__(self, library):
        targets = library.const_token
        effectors = np.concatenate([library.unary_tokens,
                                    library.binary_tokens])

        super(ConstConstraint, self).__init__(library=library,
                                              targets=targets,
                                              effectors=effectors,
                                              relationship="uchild")


class NoInputsConstraint(Constraint):
    """Class that constrains sequences without input variables.

    NOTE: This *should* be a special case of RepeatConstraint, but is not yet
    supported."""

    def __init__(self, library):
        Prior.__init__(self, library)

    def validate(self):
        if len(self.library.float_tokens) == 0:
            message = "All terminal tokens are input variables, so all" \
                "sequences will have an input variable."
            return message
        return None

    def __call__(self, actions, parent, sibling, dangling):
        # Constrain when:
        # 1) the expression would end if a terminal is chosen and
        # 2) there are no input variables
        mask = (dangling == 1) & \
               (np.sum(np.isin(actions, self.library.input_tokens), axis=1) == 0)
        prior = self.make_constraint(mask, self.library.float_tokens)
        return prior

    def is_violated(self, actions, parent, sibling):
        # Violated if no input tokens are found in actions
        tokens = self.library.input_tokens
        return bool(np.isin(tokens, actions).sum() == 0)

    def describe(self):
        message = "{}: Sequences contain at least one input variable Token.".format(self.__class__.__name__)
        return message


class InverseUnaryConstraint(Constraint):
    """Class that constrains each unary Token from being the child of its
    corresponding inverse unary Tokens."""

    def __init__(self, library):
        Prior.__init__(self, library)
        self.priors = []

        for target, effector in library.inverse_tokens.items():
            targets = [target]
            effectors = [effector]
            prior = RelationalConstraint(library=library,
                                         targets=targets,
                                         effectors=effectors,
                                         relationship="child")
            self.priors.append(prior)

    def validate(self):
        if len(self.priors) == 0:
            message = "There are no inverse unary Token pairs in the Library."
            return message
        return None

    def __call__(self, actions, parent, sibling, dangling):
        prior = sum([prior(actions, parent, sibling, dangling) for prior in self.priors])
        return prior

    def is_violated(self, actions, parent, sibling):

        for p in self.priors:
            if p.is_violated(actions, parent, sibling):
                return True

        return False

    def describe(self):
        message = [prior.describe() for prior in self.priors]
        return "\n{}: ".format(self.__class__.__name__).join(message)


class RepeatConstraint(Constraint):
    """Class that constrains Tokens to appear between a minimum and/or maximum
    number of times."""

    def __init__(self, library, tokens, min_=None, max_=None):
        """
        Parameters
        ----------
        tokens : Token or list of Tokens
            Token(s) which should, in total, occur between min_ and max_ times.

        min_ : int or None
            Minimum number of times tokens should occur.

        max_ : int or None
            Maximum number of times tokens should occur.
        """

        Prior.__init__(self, library)
        assert min_ is not None or max_ is not None, \
            "{}: At least one of (min_, max_) must not be None.".format(self.__class__.__name__)
        self.min = min_
        self.max = max_
        self.n_objects = Program.n_objects
        self.tokens = library.actionize(tokens)

        assert min_ is None, "{}: Repeat minimum constraints are not yet " \
            "supported. This requires knowledge of length constraints.".format(self.__class__.__name__)

    def __call__(self, actions, parent, sibling, dangling):
        if self.n_objects > 1:
            _, i_batch = get_position(actions, self.library.arities, n_objects=self.n_objects)
            actions = np.copy(actions)
            mask = get_mask(i_batch, actions.shape[1])
            actions[mask == 0] = -1

        counts = np.sum(np.isin(actions, self.tokens), axis=1)
        prior = self.init_zeros(actions)
        if self.min is not None:
            raise NotImplementedError
        if self.max is not None:
            mask = counts >= self.max
            prior += self.make_constraint(mask, self.tokens)

        return prior

    def is_violated(self, actions, parent, sibling):
        return bool(np.isin(actions, self.tokens).sum() > self.max)

    def describe(self):
        names = ", ".join([self.library.names[t] for t in self.tokens])
        if self.min is None:
            message = "{}: [{}] cannot occur more than {} times."\
                .format(self.__class__.__name__, names, self.max)
        elif self.max is None:
            message = "{}: [{}] must occur at least {} times."\
                .format(self.__class__.__name__, names, self.min)
        else:
            message = "{}: [{}] must occur between {} and {} times."\
                .format(self.__class__.__name__, names, self.min, self.max)
        return message


class LengthConstraint(Constraint):
    """Class that constrains the Program from falling within a minimum and/or
    maximum length"""

    def __init__(self, library, min_=None, max_=None):
        """
        Parameters
        ----------
        min_ : int or None
            Minimum length of the Program.

        max_ : int or None
            Maximum length of the Program.
        """

        Prior.__init__(self, library)
        self.min = min_
        self.max = max_
        self.n_objects = Program.n_objects
        if self.n_objects > 1:
            assert self.max is not None, "Is max length constraint turned on? Max length constraint is required when n_objects > 1."

        assert min_ is not None or max_ is not None, \
            "At least one of (min_, max_) must not be None."

    def initial_prior(self):
        prior = Prior.initial_prior(self)
        for t in self.library.terminal_tokens:
            prior[t] = -np.inf
        return prior

    def __call__(self, actions, parent, sibling, dangling):

        # Initialize the prior
        prior = self.init_zeros(actions)
        i = actions.shape[1] - 1 # Current time

        if self.n_objects > 1:
            i, _ = get_position(actions, self.library.arities, n_objects=self.n_objects)

            if self.max is not None:
                remaining = self.max - (i + 1)
                # TBD: For loop over arities
                mask = dangling >= remaining - 1 # constrain binary
                prior += self.make_constraint(mask, self.library.binary_tokens)
                mask = dangling == remaining # constrain unary
                prior += self.make_constraint(mask, self.library.unary_tokens)

            # Constrain terminals when dangling == 1 until selecting the
            # (min_length)th token
            if self.min is not None:
                mask = np.logical_and((i + 2) < self.min,
                                        dangling == 1)
                prior += self.make_constraint(mask, self.library.terminal_tokens)
        else:
            # Never need to constrain max length for first half of expression
            if self.max is not None and (i + 2) >= self.max // 2:
                remaining = self.max - (i + 1)
                # assert sum(dangling > remaining) == 0, (dangling, remaining)
                # TBD: For loop over arities
                mask = dangling >= remaining - 1 # Constrain binary
                prior += self.make_constraint(mask, self.library.binary_tokens)
                mask = dangling == remaining # Constrain unary
                prior += self.make_constraint(mask, self.library.unary_tokens)

            # Constrain terminals when dangling == 1 until selecting the
            # (min_length)th token
            if self.min is not None and (i + 2) < self.min:
                mask = dangling == 1 # Constrain terminals
                prior += self.make_constraint(mask, self.library.terminal_tokens)


        return prior

    def is_violated(self, actions, parent, sibling):
        l = len(actions[0])
        if self.min is not None and l < self.min:
            return True
        if self.max is not None and l > self.max:
            return True

        return False

    def describe(self):
        message = []
        indent = " " * len(self.__class__.__name__) + "  "
        if self.min is not None:
            message.append("{}: Sequences have minimum length {}.".format(self.__class__.__name__, self.min))
        if self.max is not None:
            message.append(indent + "Sequences have maximum length {}.".format(self.max))
        message = "\n".join(message)
        return message


class DomainRangeConstraint(Constraint):
    """
    Class that constrains two special cases regarding the domain and range of
    (X, y) data and tokens. First, for the first position, it constrains tokens
    when its range does not contain the interval [min(y), max(y)]. For example,
    if min(y) = -2, the first token cannot be sin. Second, input variable x is
    constrained if their parent would be unary and the domain of the parent
    does not contain the interval [min(x), max(x)]. For example, if min(x) = -2
    and the parent is sqrt, then x is constrained.
    """

    def __init__(self, library):
        Constraint.__init__(self, library)

        assert Program.n_objects == 1, "Not supported for n_objects > 1."

        # Min/max of dataset domain
        X_min = Program.task.X_train.min(axis=0) # (n_input_var,)
        X_max = Program.task.X_train.max(axis=0) # (n_input_var,)

        # Min/max of dataset range
        y_min = Program.task.y_train.min() # scalar
        y_max = Program.task.y_train.max() # scalar

        # TBD: Add domain/range to Token
        # Define domains of possible tokens
        domains = defaultdict(lambda : (-np.inf, np.inf))
        domains.update({
            "sqrt" : (0, np.inf),
            "log" : (0, np.inf)
        })

        # TBD: Add domain/range to Token
        # Define ranges of possible tokens
        ranges = defaultdict(lambda : (-np.inf, np.inf))
        ranges.update({
            "sin" : (-1.0, 1.0),
            "cos" : (-1.0, 1.0),
            "exp" : (0, np.inf),
            "abs" : (0, np.inf),
            "sqrt" : (0, np.inf),
            "n2" : (0, np.inf),
            "n4" : (0, np.inf),
            "n6" : (0, np.inf),
        })

        # Convert to np.ndarray of shape (L, 2)
        domains = np.array([domains[t.name] for t in self.library.tokens], dtype=np.float32)
        ranges = np.array([ranges[t.name] for t in self.library.tokens], dtype=np.float32)

        # Pre-compute the initial prior
        self.p0 = Prior.initial_prior(self) # Zeros of shape (L,)
        for i, R in enumerate(ranges):
            if y_min < R[0] or y_max > R[1]:
                self.p0[i] = -np.inf

        # Pre-compute constraining unary parents for each input variable
        self.constraining_parents = [] # List of np.ndarray
        for i in range(self.library.n_input_tokens):
            self.constraining_parents.append([])
            for t in self.library.unary_tokens:
                D = domains[t]
                if X_min[i] < D[0] or X_max[i] > D[1]:
                    adj_parent = self.library.parent_adjust[t]
                    self.constraining_parents[i].append(adj_parent)
            self.constraining_parents[i] = np.array(self.constraining_parents[i], dtype=np.int32)

        # Pre-compute logits to add when constraining maximum length and this is
        # the last chance to select a unary token (before being constrained to
        # select only terminals). This will only be used if self.max_length is later
        # set to not be None
        self.max_length = None
        self.last_chance_unary = Prior.initial_prior(self)
        for t in self.library.unary_tokens:
            parent = self.library.parent_adjust[t] 
            if all([parent in arr for arr in self.constraining_parents]):
                self.last_chance_unary[t] = -np.inf

    def initial_prior(self):
        return self.p0

    def __call__(self, actions, parent, sibling, dangling):

        prior = self.init_zeros(actions)

        # SPECIAL CASE:
        # If max length is constrained and this is the last chance to select a
        # unary token, make sure that a unary can only be chosen if it is not a
        # constraining parent of all input variables. Without this, there can be
        # a collision with LengthConstraint.
        if self.max_length is not None:
            remaining = self.max_length - actions.shape[1]
            mask = dangling == remaining - 1 # Last chance to constrain unary
            prior[mask] += self.last_chance_unary

        # For each input variable, see if the parent constrains that input
        for i, x in enumerate(self.library.input_tokens):
            logit_adjust = Prior.initial_prior(self) # Zeros of shape (L,)
            logit_adjust[x] = -np.inf
            mask = np.isin(parent, self.constraining_parents[i])
            prior[mask] += logit_adjust

        return prior

    def describe(self):
        message = []
        indent = " " * len(self.__class__.__name__) + "  "
        message.append("{}: First token's range must contain [min(y), max(y)].".format(self.__class__.__name__))
        message.append(indent + "Input variable domains must be contained in unary parent domains.")
        message = "\n".join(message)
        return message

    def validate(self):
        if self.p0.sum() == 0 and all([len(x) == 0 for x in self.constraining_parents]):
            return "All token ranges contain [min(y), max(y)] and all token domains contain [min(x), max(x)]."
        return None


class UniformArityPrior(Prior):
    """Class that puts a fixed prior on arities by transforming the initial
    distribution from uniform over tokens to uniform over arities."""

    def __init__(self, library):

        Prior.__init__(self, library)

        # For each token, subtract log(n), where n is the total number of tokens
        # in the library with the same arity as that token. This is equivalent
        # to... For each arity, subtract log(n) from tokens of that arity, where
        # n is the total number of tokens of that arity
        self.logit_adjust = np.zeros((self.L,), dtype=np.float32)
        for arity, tokens in self.library.tokens_of_arity.items():
            self.logit_adjust[tokens] -= np.log(len(tokens))

    def initial_prior(self):
        return self.logit_adjust

    def __call__(self, actions, parent, sibling, dangling):

        # This will be broadcast when added to the joint prior
        prior = self.logit_adjust
        return prior

    def describe(self):
        """Describe the Prior."""

        return "{}: Activated.".format(self.__class__.__name__)


class SoftLengthPrior(Prior):
    """Class that puts a soft prior on length. Before loc, terminal probabilities
    are scaled by exp(-(t - loc) ** 2 / (2 * scale)) where dangling == 1. After
    loc, non-terminal probabilities are scaled by that number."""

    def __init__(self, library, loc, scale):

        Prior.__init__(self, library)

        self.loc = loc
        self.scale = scale
        self.n_objects = Program.n_objects

        self.terminal_mask = np.zeros((self.L,), dtype=np.bool)
        self.terminal_mask[self.library.terminal_tokens] = True

        self.nonterminal_mask = ~self.terminal_mask

    def __call__(self, actions, parent, sibling, dangling):

        # Initialize the prior
        prior = self.init_zeros(actions)
        t = actions.shape[1] # Current time

        if self.n_objects > 1:
            t_batch, _ = get_position(actions, self.library.arities, n_objects=self.n_objects)

            # Adjust the terminal or non-terminal logits
            logit_adjust = -(t_batch - self.loc) ** 2 / (2 * self.scale)

            # Find indices where we want to decrease p(terminal). Do this when dangling == 1 and t (for that action) < loc
            idxs = np.logical_and(t_batch < self.loc,
                                  dangling == 1)
            prior[idxs] += np.outer(logit_adjust[idxs], self.terminal_mask)

            # Find indices where we want to decrease p(non-terminal). Do this when t (for that action) > loc
            nonterm_idxs = (t_batch > self.loc)
            prior[nonterm_idxs] += np.outer(logit_adjust[nonterm_idxs], self.nonterminal_mask)

        else:
            # Adjustment to terminal or non-terminal logits
            logit_adjust = -(t - self.loc) ** 2 / (2 * self.scale)

            # Before loc, decrease p(terminal) where dangling == 1
            if t < self.loc:
                prior[dangling == 1] += self.terminal_mask * logit_adjust

            # After loc, decrease p(non-terminal)
            else:
                prior += self.nonterminal_mask * logit_adjust

        return prior

    def validate(self):
        if self.loc is None or self.scale is None:
            message = "'scale' and 'loc' arguments must be specified!"
            return message
        return None


class BindingPrior(Constraint):
    """ Super class that contains shared information for both 
    SequencePositionsConstraint and Language Model-based priors. """

    def __init__(self, library, menu_file):
        """
        Parameters
        ----------       
        menu_file : str
            YAML file containing sequence positions constraints.
        """
        # lazy imports: these are specific for the binding's prior
        import yaml
        from collections import OrderedDict

        Prior.__init__(self, library)
        
        task = Program.task
        self.master_sequence = task.master_sequence # TBD: Refactor to only use master_actions
        self.master_actions = task.master_actions
        self.allowed_mutations = task.allowed_mutations
        self.mode = task.mode

        # Sequence length
        self.seq_len = len(self.master_actions)


class MutationalDistanceConstraint(BindingPrior):
    """Class the constrains the mutational distance to fall between a given
    minimum and maximum."""

    def __init__(self, library, min_, max_):

        BindingPrior.__init__(self, library, Program.task.menu_file)
        self.min = min_
        self.max = max_

        # TBD: Add support for minimum number of mutations
        if self.min is not None:
            raise NotImplementedError("Minimum number of mutations constraint \
                not yet implemented.")

        # Precompute logits for each position; used when preventing a mutation
        self.no_mut = []
        for pos in range(self.seq_len):
            pos_prior = np.full(self.L, fill_value=-np.inf, dtype=np.float32)
            pos_prior[self.master_actions[pos]] = 0
            self.no_mut.append(pos_prior)

    def __call__(self, actions, parent, sibling, dangling):

        pos = actions.shape[1] # Position along the sequence
        prior = self.init_zeros(actions)

        # TBD: Fix hack for "ending token"
        if pos == self.seq_len:
            return prior

        # Compute the number of mutations so far
        mut_dist = BU.mutational_distance(actions, 
                                          self.master_actions[:pos],
                                          aggfunc='sum')

        # Prevent further mutation when mut_dist == max
        prior[mut_dist == self.max] = self.no_mut[pos]

        return prior


class MutationProbabilityPrior(BindingPrior):
    """
    Prior that adjusts prior probabilities of mutation, independent of the
    number of allowable AAs at each position.

    This constraint handles allowable tokens, thus SequencePositionsConstraint
    is not needed when using this.

    The prior computes a desired probability vector where probability is 0 for
    disallowed AAs, p_master for the master AA, and is uniformly distributed
    across remaining AAs. If parameterized with "mut_dist" (expected mutational
    distance), then p_master = (seq_len - mut_dist) / seq_len.

    Desired probabilities for each AA are:
    1. Zero for disallowed AAs.
    2. p_master for master AA.
    3. (1 - p_master) / n for the n allowed non-master AAs.

    Parameters
    ----------
    p_mutate : float
        Probability of selecting an AA different than the master sequence.

    mut_dist : float
        Expected mutational_distance.
    """

    def __init__(self, library, p_master=None, mut_dist=None):
        BindingPrior.__init__(self, library, Program.task.menu_file)

        assert (p_master is None) + (mut_dist is None) == 1, \
            "Exactly one of [p_master, mut_dist] should be None."
        if mut_dist is not None:
            assert mut_dist < self.seq_len, "Expected mutational distance \
                must be less than sequence length."
            p_master = (self.seq_len - mut_dist) / self.seq_len

        # Precompute the prior for each position
        self.pos_prior = []
        for i in range(self.seq_len):

            # Compute allowed and master actions
            if self.mode == "short":
                pos = list(self.allowed_mutations.keys())[i]
            else:
                pos = i
            allowed_aas = self.allowed_mutations.get(pos, None)
            master_action = self.master_actions[i]

            # Position is allowed to mutate
            if allowed_aas is not None:
                allowed_actions = library.actionize(allowed_aas)

                # Probability for disallowed AAs
                p = np.zeros(self.L, dtype=np.float32)

                # Probability for allowed non-master AAs
                n = len(allowed_actions) - 1
                p_mut = (1 - p_master) / n
                p[allowed_actions] = p_mut

                # Probability for master AA
                p[master_action] = p_master

                logits = np.log(p)
                self.pos_prior.append(logits)

            # Position is not allowed to mutate
            else:
                assert self.mode == "full"
                logits = np.full(self.L, -np.inf, dtype=np.float32)
                logits[master_action] = 0
                self.pos_prior.append(logits)

    def __call__(self, actions, parent, sibling, dangling):
        pos = actions.shape[1]

        # TBD: Fix hack for "ending token"
        if pos == self.seq_len:
            return self.init_zeros(actions)

        return self.pos_prior[pos]

    def initial_prior(self):
        return self.pos_prior[0]


class SequencePositionsConstraint(BindingPrior):
    """Class that constrains Tokens to follow the constraints defined in the YAML file. """

    def __init__(self, library, biasing_factor):
        """
        Parameters
        ----------
        biasing_factor : float
            Increment factor in the prior vector pushing it towards master sequence.
        """
        BindingPrior.__init__(self, library, Program.task.menu_file)
        self.biasing_factor = float(biasing_factor)

    def initial_prior(self):
        """ Prior for time step 0 """
        prior = np.zeros((1, self.L), dtype=np.float32)
        prior = self.__calculate_prior(prior, 0)[0, :]
        return prior

    def __call__(self, actions, parent, sibling, dangling):
        """ Prior for time-step > 0. """
        # Initialize the prior
        prior = self.init_zeros(actions)
        seq_position = actions.shape[1]
        prior = self.__calculate_prior(prior, seq_position)
        return prior

    def __calculate_prior(self, prior, seq_position):
        """ Calculate prior logits based on the information in the yaml file. """
        # it's the "sequence ending" token - not going to be in the sample
        if seq_position >= len(self.master_sequence):
            return prior

        # bias sequence towards master sequence
        if self.biasing_factor > 0:
            idx = constants.AMINO_ACIDS.index(self.master_sequence[seq_position])
            prior[:, idx] = np.log(self.biasing_factor)

        if self.mode == 'short':
            # it's the "sequence ending" token - not going to be in the sample
            if seq_position >= len(self.allowed_mutations.keys()):
                return prior
            items = list(self.allowed_mutations.items())
            actual_seq_position = items[seq_position][0]
            mask = np.isin(constants.AMINO_ACIDS,
                           self.allowed_mutations[actual_seq_position],
                           invert=True)
            prior[:, mask] = -np.inf

        else:
            # check if there is any restriction for this particular position
            if seq_position in self.allowed_mutations:  # allowed to mutate
                # not all AA are allowed, but some
                if len(self.allowed_mutations[seq_position]) < len(constants.AMINO_ACIDS):
                    # False: allowed to mutate
                    # True: constrained - cannot be mutated
                    mask = np.isin(constants.AMINO_ACIDS,
                                   self.allowed_mutations[seq_position],
                                   invert=True)
                    prior[:, mask] = -np.inf
                else:
                    # all AAs have the same chance, then no constraint imposed
                    pass
            else: # mutation is not allowed
                mask = [True] * len(constants.AMINO_ACIDS)
                mask[constants.AMINO_ACIDS.index(self.master_sequence[seq_position])] = False
                prior[:, mask] = -np.inf

        return prior

    def describe(self):
        message = "Impose positional constraints (mutation allowance) in the sequence based "
        message += "on YAML file definition. Using {} sequence generation mode.".format(self.mode)
        return message


class ABAGSeqPrior(BindingPrior):
    """Wrapper class that applies a prior from the ABAGSeq Transformer model."""

    def __init__(self, library, bert_model_file):
        """
        Parameters
        ----------       
        bert_model_file : str
            Path to the pre-trained BERT model.

        """
        # lazy import: specific for the LM prior
        import torch
        from abag_seq.api.api import ABAGSequenceAPI
        from abag_seq.api.menu_manager import MenuManager

        BindingPrior.__init__(self, library, Program.task.menu_file)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lm = ABAGSequenceAPI(model_path=bert_model_file, 
                                  load_model=True,
                                  device=device)
        # menu manager handles chain type 
        self.lm_menu_manager = MenuManager(menu_fp=Program.task.menu_file)
    
    def initial_prior(self):
        # empty actions for initial priors
        priors = self.lm.batch_get_priors(actions=[[], ], orders=[[0]], manager=self.lm_menu_manager)[0].numpy()
        prior = priors[0]
        return prior
    
    def __call__(self, actions, parent, sibling, dangling):
        # order of unmasking, assumed left to right with batch
        lm_orders = [range(len(actions[0])+1)] * actions.shape[0]
        priors = self.init_zeros(actions)
        # this is necessary to handle sampling fixed positions
        if len(actions[0]) >= len(self.allowed_mutations.keys()):
            return priors
        # convert to amino acid character representation
        lm_readable_actions = [[constants.AMINO_ACIDS[aa_index] for aa_index in sample] for sample in actions]
        priors = self.lm.batch_get_priors(actions=lm_readable_actions, orders=lm_orders, manager=self.lm_menu_manager)[0].numpy()
        return priors

    def describe(self):
        message = "Encourage human-like sequences through the use of a "
        message += "Language Model (BERT) pretrained on human antibodies."
        return message


class LanguageModelPrior(Prior):
    """Class that applies a prior based on a pre-trained language model."""

    def __init__(self, library, weight=1.0, **kwargs):

        Prior.__init__(self, library)

        self.lm = LM(library, **kwargs)
        self.weight = weight

    def initial_prior(self):

        # TBD: Get initial prior from language model
        return np.zeros((self.L,), dtype=np.float32)

    def __call__(self, actions, parent, sibling, dangling):

        """
        NOTE: This assumes that the prior is always called sequentially during
        sampling. This may break if calling the prior arbitrarily.
        """
        if actions.shape[1] == 1:
            self.lm.next_state = None

        action = actions[:, -1] # Current action
        prior = self.lm.get_lm_prior(action)
        prior *= self.weight

        return prior

    def validate(self):
        if self.weight is None:
            message = "Need to specify language model arguments."
            return message
        return None


class StateCheckerConstraint(Constraint):
    """Class that impose constraints on StateChecker Tokens to avoid degenerate
    situations (e.g., checking if xi < 6 when we know xi < 3).

    Consider StateCheckers 'xi < tj' that is associated with the
    i-th state variable xi and threshold tj, and 'xl < tk' that is
    associated with the l-th state variable xl and threshold tk.

    The constraints include:
        1. 'xl < tk' cannot be the left child of 'xi < tj' if l < i or if l == i and tk >= tj
        2. 'xl < tk' cannot be the right child of 'xi < tj' if l <= i
        3. a StateChecker cannot be a child of a non-StateChecker."""

    def __init__(self, library):
        Prior.__init__(self, library)
        self.priors = []

        # Loop over StateChecker 'xi < tj'
        for parent in library.state_checker_tokens:
            effectors = [parent]
            targets = []

            # Add StateChecker 'xl < tk' to targets if l < i
            for child in library.state_checker_tokens:
                if self.library[child].state_index < self.library[parent].state_index:
                    targets.append(child)

            # Add StateChecker 'xl < tk' to targets if l == i and tk >= tj
            for child in library.state_checker_tokens:
                if self.library[child].state_index == self.library[parent].state_index:
                    if self.library[child].threshold >= self.library[parent].threshold:
                        targets.append(child)

            if len(targets) > 0:
                # Add prior that constraints targets (containing 'xl < tk' with l < i, and
                # 'xl < tk' with l == i and tk >= tj) from being the left child of 'xi < tj'
                prior = RelationalConstraint(library,
                                             targets=targets,
                                             effectors=effectors,
                                             relationship="lchild")
                self.priors.append(prior)

            # Add StateChecker 'xl < tk' to targets if l == i and tk < tj
            # (targets now contains all StateChecker 'xl < tk' with l <= i)
            for child in library.state_checker_tokens:
                if self.library[child].state_index == self.library[parent].state_index:
                    if self.library[child].threshold < self.library[parent].threshold:
                        targets.append(child)

            if len(targets) > 0:
                # Add prior that constraints targets (containing 'xl < tk' with l <= i)
                # from being the right child of 'xi < tj'
                prior = RelationalConstraint(library,
                                             targets=targets,
                                             effectors=effectors,
                                             relationship="rchild")
                self.priors.append(prior)

        # Add priors that constraint any StateChecker from being a child of any non-StateChecker
        if len(library.state_checker_tokens) > 0:
            non_state_checker_tokens = np.setdiff1d(np.arange(self.L), library.state_checker_tokens)
            for parent in non_state_checker_tokens:
                if self.library[parent].arity > 0:
                    effectors = [parent]
                    prior = RelationalConstraint(library,
                                                 targets=library.state_checker_tokens,
                                                 effectors=effectors,
                                                 relationship="child")
                    self.priors.append(prior)

    def validate(self):
        if len(self.priors) == 0:
            message = "There are no StateChecker tokens in the library."
            return message
        return None

    def __call__(self, actions, parent, sibling, dangling):
        prior = sum([prior(actions, parent, sibling, dangling)
                     for prior in self.priors])
        return prior

    def describe(self):
        indent = " " * len(self.__class__.__name__) + "  "
        message = ["StateCheckerConstraint: Sequences containing StateChecker tokens cannot produce degenerate logic."]
        message.append(indent + "'xl < tk' cannot be the left child of 'xi < tj' if l < i or if l == i and tk >= tj.")
        message.append(indent + "'xl < tk' cannot be the right child of 'xi < tj' if l <= i.")
        message.append(indent + "A StateChecker cannot be a child of a non-StateChecker.")
        return "\n".join(message)


class MutuallyExclusiveConstraint(Constraint):
    """Class that constrains the program from having two or more distinct tokens
    in a given set of tokens. Mathematically, this constrains the intersection
    of set(tokens) and set(actions) to have a resulting size of 0 or 1.
    
    For example, if the given set of tokens = ["poly", "const"], then this 
    constraint prevents actions = ["add", "const", "poly"] to be sampled. 
    Note, however, that it does not prevents the same token to appear multiple 
    times. So, e.g., actions = ["add", "poly", "poly"] is allowed."""

    def __init__(self, library, tokens):
        super().__init__(library)
        self.tokens = tokens

    def __call__(self, actions, parent, sibling, dangling):
        prior = self.init_zeros(actions)

        # For each mutually exclusive token, see if it exists.
        # If so, constrain all other mutually exclusive tokens.
        for token in self.tokens:
            mask = np.any(actions == token, axis=1)
            others = self.tokens[self.tokens != token]
            prior += self.make_constraint(mask, others)
        return prior

    def validate(self):
        if len(self.tokens) < 2:
            return "Length of {} must be at least 2".format(self.tokens)
        return None

    def is_violated(self, actions, parent, sibling):
        return np.intersect1d(self.tokens, actions).size > 1

    def describe(self):
        tokens = ", ".join([self.library.names[t] for t in self.tokens])
        message = self.__class__.__name__
        message += ": Two or more distinct tokens in [{}] cannot appear in the same sequence.".format(tokens)
        return message
                    
class PolyConstraint(Constraint):
    """Class that impose constraints such that polynomial fitting problems can be
    constructed and well-defined when Polynomial Token is in library."""
    def __init__(self, library):
        Prior.__init__(self, library)
        # poly token cannot appear more than 1 time
        self.priors = [RepeatConstraint(library, "poly", None, 1)]

        # any function whose inverse is multi-valued cannot be an ancestor of "poly"
        # because it may lead to wrong data for the polynomial fitting problem.
        invalid_ancestors = ["sin", "cos", "tan", "n2", "n4", "n6", "abs"]
        invalid_ancestors = np.intersect1d(invalid_ancestors, library.names, True)
        if len(invalid_ancestors) > 0:
            descendant_prior = RelationalConstraint(library,
                                                    targets=["poly"],
                                                    effectors=invalid_ancestors,
                                                    relationship="descendant")
            self.priors.append(descendant_prior)
        
        # poly and const cannot appear in the same traversal
        if library.const_token is not None:
            mutually_exclusive_tokens = np.array([library.poly_token, library.const_token])
            self.priors.append(MutuallyExclusiveConstraint(library, mutually_exclusive_tokens))

    def __call__(self, actions, parent, sibling, dangling):
        prior = sum([prior(actions, parent, sibling, dangling)
                     for prior in self.priors])    
        return prior

    def validate(self):
        if "poly" not in self.library.names:
            return "There is no 'poly' token in the Library"
        return None

    def is_violated(self, actions, parent, sibling):
        for prior in self.priors:
            if prior.is_violated(actions, parent, sibling):
                return True
        return False

    def describe(self):
        return "\n".join([prior.describe() for prior in self.priors])
